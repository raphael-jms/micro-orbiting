import numpy as np
import casadi as ca
import math
import matplotlib.pyplot as plt
# import micro_orbiting_mpc.models.ff_input_bounds as ff_bounds
from micro_orbiting_mpc.models.ff_input_bounds import PlottingHelper, SpiralParameters
import copy


class FreeFlyerDynamicsSimplified:
    def __init__(self, dt) -> None:
        """
        Class implementing the dynamics of the robot using the simplified dynamics that
        disregard the physical inputs and see the robot as driven by two forces and a torque. 
        Adapted from https://github.com/KTH-DHSG/corridor_mpc/tree/main

        :param mass: mass of the robot
        :type mass: float
        :param J: inertia matrix of the robot
        :type J: np.array
        :param max_force: maximum force that one thruster can apply
        :type max_force: float
        :param dt: time step of the simulation
        :type dt: float
        :param n: number of states
        :type n: int
        :param m: number of inputs for the generalized input
        :type m: int
        :param m_full: number of inputs for the full physical input
        :type m_full: int
        :param faulty_input_full: faulty input for the full physical input in N
        :type faulty_input_full: np.array
        :param faulty_input_simple: faulty input for the simplified resulting Force/Torque
                 input in N resp. Nm
        :type faulty_input_simple: np.array
        """

        self.dt = dt 

        self.n = 6
        self.m_simplified = 3
        self.m_full = 8

        self.faulty_input_full = np.zeros((self.m_full, 1))
        self.faulty_input_simple = np.zeros((self.m_simplified, 1))

        self.set_system_constants()
        self.set_casadi_options()
        if type(self) == FreeFlyerDynamicsSimplified:
            # Child classes should set the dynamics themselves
            self.set_dynamics()

        self.name="FreeFlyerDynamicsSimplified"

        # The model itself calculates with the physical values. In order to get cm-precision,
        # convert into the appropriate range of values for the error function, eg [x1]=1m=100cm
        #                 in the model:   m,   m, rad, m/s, m/s, rad/s 
        self.normalize_error = np.array([100, 100,   1, 100, 100, 1])

    def set_dynamics(self):
        self.dynamics = self.rk4_integrator(self.dynamics_simplified)

    def set_system_constants(self):
        """
        Set the constants of the system.
        """

        self.mass = 14.5
        self.J = 0.370
        self.max_force = 1.75

        self.u_lb_physical = np.zeros((self.m_full, 1))
        self.u_ub_physical = self.max_force * np.ones((self.m_full, 1))

        # distance of the thrusters to the center of mass
        self.d = 0.14
        d = self.d

        # Ref. comment in spacecraft_mpc_node / publish_control: As Gazebo is using NWU instead of 
        # ENU, the axes need to be turned from the original position

        # Matrix to calculate resulting forces and torques from the thruster forces
        self.u_full2u_simple_np = np.array([[ 1, -1,  1, -1,  0,  0,  0,  0],
                                         [ 0,  0,  0,  0,  1, -1,  1, -1],
                                         [ d, -d, -d,  d,  d, -d, -d,  d]])

        # # Matrix to calculate resulting forces and torques from the thruster forces
        # self.u_full2u_simple_np = np.array([
        #                                  [ 1, -1,  1, -1,  0,  0,  0,  0],
        #                                  [ 0,  0,  0,  0, -1,  1, -1,  1],
        #                                  [-d,  d,  d, -d, -d,  d,  d, -d]])

        self.u_full2u_simple = ca.MX(self.u_full2u_simple_np)

        self.u_ub_simple = np.array([2*self.max_force, 2*self.max_force, 4*self.max_force*d])
        self.u_lb_simple = np.array([-2*self.max_force, -2*self.max_force, -4*self.max_force*d])

        # TODO This is not quite correct, the actual bounds can be found in InputHandler.
        self.uub = self.u_ub_simple
        self.ulb = self.u_lb_simple

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def dynamics_simplified(self, x, u):
        """
        Dynamics of the robot.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX

        The state is defined as:
        [x, y, alpha, vx, vy, omega]

        The simplified control input is defined as:
        [fx, fy, tau]
        """
        # Unpack state
        p = x[0:3]
        v = x[3:6]

        # Control input
        f = u + self.faulty_input_simple

        # Velocity
        pdot = v

        # Rotation matrix
        R = ca.MX.zeros(2, 2)
        R[0, 0] = ca.cos(p[2])
        R[0, 1] = -ca.sin(p[2])
        R[1, 0] = ca.sin(p[2])
        R[1, 1] = ca.cos(p[2])

        # Acceleration
        vdot_lin = (1 / self.mass) * R @ f[0:2]
        vdot_rot = (1 / self.J) * f[2]

        return ca.vertcat(*[pdot, vdot_lin, vdot_rot])

    def add_actuator_fault(self, pos, percentage):
        """
        Set the constant actuator fault. 

        :param pos: position of the actuator
        :type pos: [int, int]
                   with pos=[i,j] means thruster u_{i,j}
                   the thrusters are ordered as u = [u11, u12, u21, u22, u31, u32, u41, u42]
        :param percentage: value of the actuator fault (0-1)
        :type percentage: float
        """
        idx = (pos[0]-1)*2 + (pos[1]-1)

        self.u_ub_physical[idx] = 0

        self.faulty_input_full[idx] = self.max_force * percentage
        self.faulty_input_simple = self.u_full2u_simple_np @ self.faulty_input_full
        
        '''
        Since self.dynamics is a casadi function, it is actually an immutable object.
        Modifying its parameters means we have to recreate the object again (/bc of immutable).
        Same for all objects that directly use it!!! 
        (see SimEnvironment.set_actuator_fault)
        '''
        self.set_dynamics()
    
    def add_actuator_fault_from_fault_vector(self, faulty_input, u_ub_physical):
        """
        Set the constant actuator fault from a vector of faulty inputs. 

        :param faulty_input: The fault value in Newton; full/physical input
        :type faulty_input: numpy array
        :param u_ub_physical: The upper bound for the input of a F_{i,j}
        :type u_ub_physical: numpy array
        """
        for i, max_input in enumerate(u_ub_physical):
            if max_input == 0:
                # Faulty inputs are the ones where there is no max. (controllable) input
                # Get the two indices for add_actuator_fault
                idxs = [math.floor(i/2) + 1, i%2 + 1]
                self.add_actuator_fault(idxs, faulty_input[i]/self.max_force)

    def set_dynamics(self):
        self.dynamics = self.rk4_integrator(self.dynamics_simplified)

    def get_faulty_input(self):
        return self.faulty_input_simple, self.faulty_input_full

    def copy_new_dt(self, dt):
        """
        Clone the object with a new time step.

        :param dt: new time step
        :type dt: float
        :return: new object
        :rtype: FreeFlyerDynamicsSimplified
        """
        if self.dt == dt:
            return self
        
        new_obj = copy.deepcopy(self)
        new_obj.dt = dt
        new_obj.set_dynamics()

        return new_obj

    @property
    def m(self):
        return self.m_simplified

class FreeFlyerDynamicsFull(FreeFlyerDynamicsSimplified):
    def __init__(self, dt) -> None:
        """
        Class implementing the actual physical dynamics of the robot with the pysical inputs

        :param max_force: maximum force that one thruster can apply
        :type max_force: float
        :param dt: time step of the simulation
        :type dt: float
        :param m_simplified: number of inputs for the generalized input
        :type m_simplified: int
        :param m_full: number of inputs for the full physical input
        :type m_full: int
        :param chull_input_bounds: bounds for the generalized input forces
        :type chull_input_bounds: scipy.spatial ConvexHull
        """

        FreeFlyerDynamicsSimplified.__init__(self, dt)

        self.uub = self.u_ub_physical
        self.ulb = self.u_lb_physical

        self.set_dynamics()

        self.name="FreeFlyerDynamicsFull"

    def dynamics_full(self, x, u_full):
        """
        The dynamics considering the actual physical inputs to the system.

        u_full = [u11, u12, u21, u22, u31, u32, u41, u42]
        u_simple = [fx, fy, tau]
        """
        u_simple = self.u_full2u_simple @ u_full
        return self.dynamics_simplified(x, u_simple)

    def set_dynamics(self):
        self.dynamics = self.rk4_integrator(self.dynamics_full)

    def add_actuator_fault(self, pos, percentage):
        super().add_actuator_fault(pos, percentage)
        self.uub = self.u_ub_physical
        self.ulb = self.u_lb_physical

    @property
    def m(self):
        return self.m_full

class SpiralDynamics(FreeFlyerDynamicsSimplified):
    def __init__(self, dt, spiral_params) -> None:
        """
        Class implementing the dynamics of the FreeFlyer w.r.t. the center point of the spiral.
        The simulator should never use this model, it is only used for the MPC. Instead, the
        FreeFlyerDynamicsFull should be used for the simulator.
        """
        self.r = spiral_params.r
        self.omega_theta = spiral_params.omega_theta
        self.b = spiral_params.b
        self.spiral_params = spiral_params

        super().__init__(dt)

        self.normalize_error = np.ones(5)
        self.set_dynamics()

        self.name = "SpiralDynamics"

    @classmethod
    def from_ff_model(cls, ff_model):
        """
        Create a new object from an existing FreeFlyerDynamicsFull object.
        """
        spiral_params = SpiralParameters(ff_model)
        new_spiral_model = cls(ff_model.dt, spiral_params)
        new_spiral_model.add_actuator_fault_from_fault_vector(ff_model.faulty_input_full, 
                                                              ff_model.u_ub_physical)
        return new_spiral_model
    
    def set_dynamics(self):
        self.dynamics = self.rk4_integrator(self.spiral_dynamics)

    def spiral_dynamics(self, c, u):
        """
        Dynamics of the center point of the spiral.

        :param c: state of the center point 
        :type c: ca.MX
        :param u: control input
        :type u: ca.MX 3x1
        :return: state time derivative
        :rtype: ca.MX

        The state is defined as:
        [c1, c2, c3, c4, omega, alpha]
         0   1   2   3   4      5

        The control input is defined as: u_{ij}
        """
        alpha = c[5]
        omega = c[4]

        R = ca.MX.zeros(3, 3)
        R[0, 0] = ca.cos(alpha)
        R[0, 1] = -ca.sin(alpha)
        R[1, 0] = ca.sin(alpha)
        R[1, 1] = ca.cos(alpha)
        R[2, 2] = 1

        # Rinv = ca.MX.zeros(3, 3)
        # Rinv[0, 0] = ca.cos(alpha)
        # Rinv[0, 1] = ca.sin(alpha)
        # Rinv[1, 0] = -ca.sin(alpha)
        # Rinv[1, 1] = ca.cos(alpha)
        # Rinv[2, 2] = 1

        [Fx_res, Fy_res, T_res] = ca.vertsplit(u + self.faulty_input_simple)

        c_vdot = R @ ca.vertcat(Fx_res/self.mass - T_res*self.r/self.J, 
                                Fy_res/self.mass - self.r * omega**2, 
                                T_res/self.J)
        
        return ca.vertcat(*[c[2:4], c_vdot, omega])

    def robot_to_center(self, x):
        """
        Transform the state of the robot to the state of the center point of the spiral.
        """
        alpha = x[2]
        omega = x[5]

        c1 = x[0] - self.r * ca.sin(alpha)
        c2 = x[1] + self.r * ca.cos(alpha)
        c3 = x[3] - self.r * omega * ca.cos(alpha)
        c4 = x[4] - self.r * omega * ca.sin(alpha)

        return ca.vertcat(c1, c2, c3, c4, omega, alpha)

    def center_to_robot(self, c):
        """
        Transform back from the center to the robot
        """
        alpha = c[5]
        omega = c[4]

        x1 = c[0] + self.r * ca.sin(alpha)
        y1 = c[1] - self.r * ca.cos(alpha)
        x2 = c[2] + self.r * omega * ca.cos(alpha)
        y2 = c[3] + self.r * omega * ca.sin(alpha)

        return ca.vertcat(x1, y1, alpha, x2, y2, omega)
