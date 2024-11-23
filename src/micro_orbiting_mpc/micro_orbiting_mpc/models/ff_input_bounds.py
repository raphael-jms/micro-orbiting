import numpy as np
from itertools import product
from math import sin, cos, pi, atan2
import math
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, HalfspaceIntersection
import polytope
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp
from qpsolvers import solve_qp
import osqp
# import cdd as pcdd
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from itertools import combinations, product, permutations

# import util.animate as visuals
from controllers.state_constants import States

class Ellipse:
    def __init__(self, B, d) -> None:
        """
        Ellipse of the form

            x <= B u + d 
        
        with ||u|| <= 1.

        :param ellipsoid: Ellipoid in matrix form
        :type ellipsoid: dict{B: np.array 3x3, d: np.array 3x1}
        """
        self.B = B
        self.d = d

    def __str__(self) -> str:
        return f"B: {self.B}, d: {self.d}"

class InputBounds:
    def __init__(self, model) -> None:
        """
        Containing the bounds on the individual actors as well as the bounds on the
        resulting forces and torque
        """
        # The blocking state of the actuators F_{i}
        self.blocking_state = States.INVALID * np.ones(4)
        
        # The bounds on F_{i}
        self.Fplus = np.zeros(4)
        self.Fminus = np.zeros(4)

        # Bounds of the polytope restricting the generalized input
        self.vertices = None
        self.conv_hull = None

        # Calculate the values for the current model
        self.calculate_state(model)

    def calculate_state(self, model):
        self.model = model
        self.faulty_input_simple = model.faulty_input_simple
        self.faulty_input_full = model.faulty_input_full

        self._calc_blocking_state()
        self._calc_F_i_bounds()
        self._calc_conv_hull()

    def _calc_blocking_state(self):
        """
        Calculate the blocking state of the actuators F_{i}. The blocking state describes how 
        the two opposing actors F_{i,1} and F_{i,2} behave. Save in self.blocking_state.
        """
        for i in range(4):
            if self.model.u_ub_physical[2*i] != 0 and self.model.u_ub_physical[2*i+1] != 0:
                self.blocking_state[i] = States.OPPOSING_FREE
            elif self.model.u_ub_physical[2*i] == 0 and self.model.u_ub_physical[2*i+1] != 0:
                self.blocking_state[i] = States.FIRST_STUCK
            elif self.model.u_ub_physical[2*i] != 0 and self.model.u_ub_physical[2*i+1] == 0:
                self.blocking_state[i] = States.SECOND_STUCK
            elif self.model.u_ub_physical[2*i] == 0 and self.model.u_ub_physical[2*i+1] == 0:
                self.blocking_state[i] = States.BOTH_STUCK
            else:
                raise ValueError("This case should never happen.")

    def _calc_F_i_bounds(self):
        """
        Calculate the helper variables F_{i+} and F_{i-} for the input bounds.
        """
        if np.any(self.blocking_state == States.INVALID):
            self._calc_blocking_state()

        for i in range(4):
            # self.print_blocking_state(self.blocking_state[i])
            match self.blocking_state[i]:
                case States.OPPOSING_FREE:
                    self.Fminus[i] = -self.model.max_force
                    self.Fplus[i] = self.model.max_force
                case States.FIRST_STUCK:
                    self.Fminus[i] = -self.model.max_force + self.model.faulty_input_full[2*i]
                    self.Fplus[i] = self.model.faulty_input_full[2*i]
                case States.SECOND_STUCK:
                    self.Fminus[i] = -self.model.faulty_input_full[2*i+1]
                    self.Fplus[i] = self.model.max_force - self.model.faulty_input_full[2*i+1]
                case States.BOTH_STUCK:
                    self.Fminus[i] = self.model.faulty_input_full[2*i] - self.model.faulty_input_full[2*i+1]
                    self.Fplus[i] = self.Fminus[i]

    def _calc_conv_hull(self):
        """
        Takes the min and max values from the actual actuators and translates them into
        maximal resulting forces and torque

        Gives back a list of all possible vertices of the convex hull and the convex hull itself
        """
        # max forces in percent
        min_values = self.Fminus/self.model.max_force
        max_values = self.Fplus/self.model.max_force

        D = self.model.max_force*np.array([[1,1,0,0],[0,0,1,1]])
        d = self.model.d # distance of thrusters and center of mass
        L = self.model.max_force*np.array([d,-d,d,-d])

        list_of_input_forces = list(product(
            [min_values[0],max_values[0]],
            [min_values[1],max_values[1]],
            [min_values[2],max_values[2]],
            [min_values[3],max_values[3]]
        ))

        vertices = [] # resulting generalized force (force+torque)
        for input_force in list_of_input_forces:
            # res_f = np.matmul(np.matmul(Lambda, D), np.array(input_force))
            res_f = np.matmul(D, np.array(input_force))
            res_t = np.matmul(L, np.array(input_force))
            vertices.append(np.append(res_f, res_t))

        self.vertices = np.array(vertices)
        self.conv_hull = ConvexHull(self.vertices)

        return self.vertices, self.conv_hull

    def get_blocking_state(self, i):
        """
        Get the blocking state of the i-th actuator F_{i}.
        """
        return self.blocking_state[i]

    def print_blocking_state(self, state):
        match state:
            case States.OPPOSING_FREE:
                print("OPPOSING_FREE")
            case States.FIRST_STUCK:
                print("FIRST_STUCK")
            case States.SECOND_STUCK:
                print("SECOND_STUCK")
            case States.BOTH_STUCK:
                print("BOTH_STUCK")
            case _:
                print("Unknown state")

    def get_F_i_bounds(self):
        """
        Returns the bounds on the forces that result by summing up to opposite actuators.
        I.a. F_{i} = u_{i,1} - u_{i,2} \in [F_{i+}, F_{i-}]
        """
        if self.Fplus is None or self.Fminus is None:
            self._calc_F_i_bounds()
        return self.Fplus, self.Fminus

    def get_conv_hull(self):
        """
        Gives back the convex hull of the possible resulting forces and torques that can be 
        applied. These are calculated based on the uncontrollable force of the model.

        :return: Matrix A and vector b of the convex hull
        :rtype: np.array, np.array
        """
        if self.conv_hull is None:
            self._calc_conv_hull()

        A = self.conv_hull.equations[:, :-1]
        b = -self.conv_hull.equations[:, -1]
        return A, b

    def get_max_forces(self):
        return self.vertices

class InputHandlerImproved:
    def __init__(self, model, bounds) -> None:
        """
        Class that serves the purpose of
        - clipping the generalized input to the physical limits / handling the constraints on 
          the generalized input
        - calculating the physical input from the simplified input.
        Both is achieved when only calling get_physical_input().

        :param model: Model of the system
        :type model: FreeFlyerDynamics
        :param bounds: Bounds on the input
        :type bounds: InputBounds
        """
        self.model = model
        self.faulty_input_simple = model.faulty_input_simple
        self.faulty_input_full = model.faulty_input_full
        self.input_bounds = bounds

        self.all_c_diff = []

    def clip_generalized_input(self, u):
        """
        Clip the generalized input to the physical limits.

        :param u: generalized input
        :type u: np.array
        :return: clipped generalized input
        :rtype: np.array

        The optimal way of doing this is to optimize

        min(u_clipped) (u - u_clipped)^2
        s.t. A u_clipped <= b

        using a QP solver.

        Some (very simple) benchmarking resulted in finding a solution for the QP in 0.000775s
        which is acceptable. It might even be better to pass on a check if the input is within 
        the bounds, depending on how often the bounds are violated.
        """
        A, b = self.input_bounds.get_conv_hull()

        if np.all(A @ u <= b):
            # constraints satisfied
            return u

        return solve_qp(np.identity(3), -u, A, b, solver="daqp")

    def get_physical_input(self, u_simple):
        """
        Get the physical input from the simplified input. The function assumes that the 
        desired input is within the physical limits.

        :param u_simple: simplified input
        :type u_simple: np.array
        :return: generalized input
        :rtype: np.array

        The bounds for the force that two opposing thrusters u_{i,1} and u_{i,2} can apply are:
        F_{i-} = -u_{i,2,max} and
        F_{i+} = u_{i,1,max}
        """
        # The following code gives the actual inputs that are needed if the _resulting force_
        # is given, i.e. controlled _and_ uncontrolled (faulty) input!
        u_simple += self.faulty_input_simple.flatten()
        u_simple = self.clip_generalized_input(u_simple)
        # print(f"u_simple: {u_simple}")
        # print(f"0.9*u_simple: {0.9*u_simple}")

        Fx = u_simple[0]
        Fy = u_simple[1]
        T  = u_simple[2] / self.model.d # see derivation, the torque is divided by its lever arm
        T_tilde = (Fx + Fy + T)/2

        Fplus, Fminus = self.input_bounds.get_F_i_bounds()

        c_min = max(
            T_tilde - Fplus[0],
            - Fx + T_tilde + Fminus[1],
            Fminus[2],
            Fy - Fplus[3]
        )
        c_max = min(
            T_tilde - Fminus[0],
            - Fx + T_tilde + Fplus[1],
            Fplus[2],
            Fy - Fminus[3]
        )

        if abs(c_min-c_max) < 0.05: # We don't let us stop by numerical inaccuracies :)
            c = c_min
        else:
            if c_min > c_max:
                # print("\n++++++++++++")
                print(f"Fplus: {Fplus.T}")
                print(f"Fminus: {Fminus.T}")
                print(f"u_simple: {u_simple}")
                print(f"Fx: {Fx}, Fy: {Fy}, T: {T}, T_tilde: {T_tilde}")
                print(f"c_opt: {(2*Fy + T)/4}, cm: {c_min}, cM: {c_max}, cdiff: {c_max-c_min}")
                print("elements cmin")
                print(f"{Fx - Fplus[0]}\t{- Fx + T_tilde + Fminus[1]}\t{Fminus[2]}\t{Fy-Fplus[3]}")
                print("elements cmax")
                print(f"{Fx - Fminus[0]}\t{- Fx + T_tilde + Fplus[1]}\t{Fplus[2]}\t{Fy-Fminus[3]}")
                # print("++++++++++++\n")
                
                chull = self.input_bounds.get_conv_hull()
                max_f = self.input_bounds.get_max_forces()
                PlottingHelper.plot_convex_hull(chull, max_f, [u_simple, 
                                            self.faulty_input_simple.flatten(),
                                            u_simple - self.faulty_input_simple.flatten()])

                raise ValueError("Warning: Infeasible generalized input")

            c = np.clip( (2*Fy + T)/4, c_min, c_max).item()

        u1_tilde = T_tilde - c
        u2_tilde = Fx - T_tilde + c
        u3_tilde = c
        u4_tilde = Fy - c

        uij = []

        for i, u_tilde in enumerate([u1_tilde, u2_tilde, u3_tilde, u4_tilde]):
            match self.input_bounds.get_blocking_state(i):
                case States.OPPOSING_FREE:
                    uij += [abs(u_tilde), 0] if u_tilde >= 0 else [0, abs(u_tilde)]
                case States.FIRST_STUCK:
                    const_input = self.faulty_input_full[2*i].item()
                    uij += [0, const_input - abs(u_tilde)]
                case States.SECOND_STUCK:
                    const_input = self.faulty_input_full[2*i + 1].item()
                    uij += [const_input + abs(u_tilde), 0]
                case States.BOTH_STUCK:
                    uij += [0, 0]

        self.all_c_diff.append([c_max, c_min]) # logging functionality

        return np.array(uij).flatten()

    def plot_c_bounds(self):
        if self.c_bounds_used():
            fig2, axs2 = plt.subplots(1, 1)
            allc = np.array(self.all_c_diff)
            axs2.plot(allc[:,0], label='c_max')
            axs2.plot(allc[:,1], label='c_min')
            axs2.plot(allc[:,0] - allc[:,1], label='c_diff')
            axs2.grid(True)
            axs2.legend()

    def c_bounds_used(self):
        """
        Returns if the method get_physical_input() has been called and the bounds on c have 
        been stored.
        """
        return self.all_c_diff != []

    # def visualize_physical_input(self, u):
    #     """
    #     Visualize the inputs that can be calculated with get_physical_input().
    #     Shows both the physical inputs as well as the resulting force.

    #     :param u: generalized input
    #     :type u: np.array 8x1
    #     """
    #     visualizer = visuals.ForceVisualizer(self.model)
    #     visualizer.plot(u)

    def get_fault_category(self):
        """
        Get the fault category that occured. Returned as class constants as defined in the
        beginning of the class.
        """
        # First check if the origin is outside of the convex hull
        A, b = self.input_bounds.get_conv_hull()
        condition_vector = A @ np.zeros(self.model.m_simplified) - b

        if np.any(condition_vector > 0):
            # condition A u <= b is not satisfied for u=0
            return States.ORIGIN_OUTSIDE_OF_U
        elif np.all(condition_vector < 0):
            # condition A u <= b is satisfied without equality, i.e. origin in interior
            return States.ORIGIN_IN_INTERIOR_OF_U
        else:
            return States.ORIGIN_ON_BOUNDARY_OF_U

class PlottingHelper:
    def _plot_planes(points, ax, alpha_val=0.1):
        '''
        Plot helpers that make the plots easier to interpret
        '''
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Create a meshgrid for each plane
        xy_plane_x, xy_plane_y = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
        yz_plane_y, yz_plane_z = np.meshgrid(np.linspace(min(y), max(y), 10), np.linspace(min(z), max(z), 10))
        zx_plane_x, zx_plane_z = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(z), max(z), 10))

        ax.plot_surface(xy_plane_x, xy_plane_y, np.zeros_like(xy_plane_x), alpha=alpha_val)
        ax.plot_surface(np.zeros_like(yz_plane_y), yz_plane_y, yz_plane_z, alpha=alpha_val)
        ax.plot_surface(zx_plane_x, np.zeros_like(zx_plane_x), zx_plane_z, alpha=alpha_val)

        ax.plot([min(x), max(x)], [0,0], [0,0], "k", alpha=0.8)
        ax.plot([0,0], [min(y), max(y)], [0,0], "k", alpha=0.8)
        ax.plot([0,0], [0,0], [min(z), max(z)], "k", alpha=0.8)

    def _plot_convex_hull_from_vertices(conv_hull, vertices, ax, plotcmd="r-"):
        '''
        Calculates the convex hull from a list of vertices and plots it in 3D
        '''
        # Plot defining corner points
        ax.plot(vertices.T[0], vertices.T[1], vertices.T[2], "ko")

        for s in conv_hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(vertices[s, 0], vertices[s, 1], vertices[s, 2], plotcmd)

    def plot_convex_hull_from_inequality(A, b, *args, **kwargs):
        """
        Calculate the vertices of the polytope described by Ax <= b, then use the usual 
        function
        """
        poly = polytope.Polytope(A, b)
        feasible_point = poly.chebXc
        conv_hull = np.concatenate((A, -b[:, None]), axis=1)
        # Find the vertices of the polytope
        hs = HalfspaceIntersection(conv_hull, feasible_point)
        vertices = hs.intersections
        # print(vertices)
        conv_hull = ConvexHull(vertices)

        PlottingHelper.plot_convex_hull(conv_hull, vertices, *args, **kwargs)

    def plot_ellipsiod(ellipsoid, ax):
        """
        Plot the ellipsoid that is inscribed in the convex hull on the given axis.

        The ellipsoid is parametrized by x = Bu + d with ||u|| <= 1
        """
        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Create a meshgrid of a ball with radius 1
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                # Transform the ball to the ellipsoid
                [x[i,j],y[i,j],z[i,j]] = ellipsoid.B @ [x[i,j],y[i,j],z[i,j]] + ellipsoid.d
        
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
        ax.plot([ellipsoid.d[0]], [ellipsoid.d[1]], [ellipsoid.d[2]], "bo", alpha=0.2)
        ax.set_aspect('equal')

    def plot_convex_hull(conv_hull, max_forces, des_force=None, ellipsoid=None, ax=None, color="k",
                         title="Bounds for resulting input"):
        '''
        Plots the convex hull of the possible forces and torques that can be applied. If a 
        desired force is given, it will be plotted as well.

        :param des_force: Desired force that should be applied (multiple can be given)
        :type des_force: list of np.array
        '''
        show = ax is None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        PlottingHelper._plot_convex_hull_from_vertices(conv_hull, max_forces, ax, color)
        PlottingHelper._plot_planes(max_forces, ax)

        colors=['blue', 'green', 'purple', 'orange', 'red', 'black', 'yellow', 'pink', 'brown']
        if des_force is not None:
            for i, force in enumerate(des_force):
                [Fx, Fy, T] = force
                plt.quiver([0], [0], [0], [Fx], [Fy], [T], color=colors[i])
        
        if ellipsoid is not None:
            PlottingHelper.plot_ellipsiod(ellipsoid, ax)
        
        # plt.xlabel("Fx")
        # plt.ylabel("Fy")
        # ax.set_zlabel("T")
        plt.xlabel("$F_x$")
        plt.ylabel("$F_y$")
        ax.set_zlabel("$T$")

        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    def plot_spiral_params(spiral_params):
        """
        Convencience function to call plot_convex_hull with the parameters of the spiral path
        """
        ellipse = spiral_params.get_ellipsoid_in_conv_hull(*spiral_params.input_bounds.get_conv_hull())
        return PlottingHelper.plot_convex_hull(spiral_params.input_bounds.conv_hull,
                                        spiral_params.input_bounds.vertices,
                                        [spiral_params.faulty_input_simple.flatten(),
                                         spiral_params.b],
                                        ellipse)

class SpiralParameters:
    def __init__(self, model) -> None:
        """
        Class calculating the necessary parameters for the spiral path.
        """
        self.model = model
        self.mass = model.mass
        # The physical constant error
        self.faulty_input_simple = model.faulty_input_simple.flatten()
        self.faulty_input_full = model.faulty_input_full.flatten()

        self.input_bounds = InputBounds(model)
        self.input_handler = InputHandlerImproved(model, self.input_bounds)

        ''' 
        By the constraints, the matrices will have the form
        B = [i, 0, 0]
            [0, i, 0]
            [0, 0, j]
        d = [k, l, 0]
        '''
        ellipse = self.get_ellipsoid_in_conv_hull(*self.input_bounds.get_conv_hull())
        # Not the physical constant error, but where we would like to have it
        # The virtual constant error, so to speak
        self.b = ellipse.d 
        self.b[2] = 0 # due to numerical optimization, this can be slightly off
        self.b_norm = np.linalg.norm(self.b)

        # The control bounds given the virtual uncontrollable force; an ellipse
        self.control_bound_under_b = np.min(np.diag(ellipse.B))
        # The force to compensate the physical uncontrollable force & get the virtual one
        self.compensation_force = self.b - self.faulty_input_simple

        # The angle of attack of b w.r.t. the local coordinate system
        self.beta = atan2(self.b[1], self.b[0])

        # Parameters of the spiral path
        # self.omega_theta = 1.5
        self.omega_theta = 0.5
        self.r = self.b_norm / (self.mass * self.omega_theta**2)

    def get_ellipsoid_in_conv_hull(self, A, b):
        """
        Get maximum volume inscribed ~ellipsoid~ circle, i.e. the point within the conv hull 
        with the biggest distance to all bounds
        Adapted from https://web.stanford.edu/%7Eboyd/cvxbook/bv_cvxbook.pdf p. 414

        Constraints:
        (1) The convex hull
        (2) Fx and Fy should have same distance to the boundary 
            -> For the spiral path, this is necessary
        (3) T = 0
            -> full compensation of the uncontrollable torque
        (4) The ellipsoid is not rotated (B is identity matrix)
        """
        n = 3
        # variables
        Bvar = cp.Variable((n, n), symmetric=True)
        dvar = cp.Variable(n)
        # constraints: first conv hull constraints
        constraints = []
        for i in range(len(b)):
            constraints.append(
                cp.norm(Bvar @ A[i,:]) + A[i,:] @ dvar <= b[i]
            )
        # constraints: F and Fy same range
        constraints.append( Bvar[0,0]==Bvar[1,1])
        constraints.append( Bvar[2,2]==Bvar[1,1])
        # constraints: T = 0
        constraints.append(
            dvar[2]==0
        )
        # constraints: Bvar is identity matrix
        constraints.append( Bvar[0,1]==0 )
        constraints.append( Bvar[0,2]==0 )
        constraints.append( Bvar[1,2]==0 )
        # objective (log_det should implicitly ensure positive definiteness)
        objective = cp.Minimize(-cp.log_det(Bvar))
        # problem
        problem = cp.Problem(objective, constraints)
        # solve problem
        x = problem.solve(solver=cp.SCS)
        # solutions
        variables = problem.solution.primal_vars

        return Ellipse(Bvar.value, dvar.value)

    def print(self):
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Overview over errors and constants:")
        print(f"Generalized error: {self.faulty_input_simple.T}")
        print(f"Physical error: {self.faulty_input_full.T}")
        print(f"Virtual error b: {self.b.T} with norm {self.b_norm}")
        print(f"Spiral constants: Radius: {self.r}, omega_theta: {self.omega_theta}")
        print(f"Linear control input constraints: {self.control_bound_under_b} (radius around b)")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

class TrajectoryModifier:
    def __init__(self, trajectory, spiral_params) -> None:
        """
        Class that modifies the trajectory to be followed by the robot. The trajectory is
        modified to suit the prerequsites that a fault in the system poses to the trajectory
        and the controller that is following it.
        """
        self.original_trajectory = trajectory.copy()
        self.max_acceleration = None
        self.scaled_trajectory = None
        self.dt = spiral_params.model.dt

        self.modify_trajectory(spiral_params)
    
    def modify_trajectory(self, spiral_params):
        """
        Modify the given trajectory to fit the new controller. Modifies the whole trajectory, 
        i.e. assumes that the robot is at the start of the trajectory
        """
        # Calculate the maximum forces that are required by the old trajectory
        # Calculate the 2nd derivative of the trajectory
        x = self.original_trajectory[0:2, :]
        v = self.original_trajectory[3:5, :]

        # Unsure which one is better: As long as the velocities and positions match, it is F1,
        # if not F2 might be better
        # F1 = np.gradient(v, axis=1) / self.model.dt * self.model.mass
        # F2 = np.diff(x, n=2, axis=1) / self.model.dt**2 * self.model.mass
        m_dotdot_v1 = np.gradient(v, axis=1) / self.dt
        m_dotdot_v2 = np.diff(x, n=2, axis=1) / self.dt**2
        m_dotdot_v3 = np.gradient(np.gradient(x, axis=1), axis=1) / self.dt**2

        m_dotdot = m_dotdot_v3
        # maximal acceleration required by the trajectory
        m_dotdot_abs = np.max(np.multiply(m_dotdot, m_dotdot))
        self.max_acceleration = m_dotdot_abs

        r_phi = spiral_params.control_bound_under_b
        print(f"r_phi: {r_phi}")
        b = spiral_params.b
        print(f"b: {b}")
        print(f"abs max. acceleration: {m_dotdot_abs}")

        a = math.pow(r_phi / m_dotdot_abs, 0.25) # scaling factor for speed of the trajectory

        if a >= 1:
            # i.e. the bounds are satisfied without changing the trajectory
            self.scaled_trajectory = self.original_trajectory
            print(f"Original trajectory is feasible with a = {a}")
            return self.scaled_trajectory

        # every value needs to appear n times:
        n = math.ceil(1/a)

        """
        The below code gives rather rough results. Instead, an interpolating method could be 
        written that drops the original values completely and interpolates with a dt that is
        not a multiple of the original dt. 
        """
        new_traj = np.zeros((self.original_trajectory.shape[0], 
                             (self.original_trajectory.shape[1] - 1) * n + 1 ))
        for i in range(self.original_trajectory.shape[1]-1):
            diff = self.original_trajectory[:, i+1] - self.original_trajectory[:, i]
            for j in range(n):
                new_traj[:, i*n + j] = self.original_trajectory[:, i] + j/n * diff
        new_traj[:, -1] = self.original_trajectory[:, -1]
        # scale the velocities: e.g. twice as many points means half the speed
        new_traj[3:6, :] = new_traj[3:6, :] / n

        self.scaled_trajectory = new_traj

        print(f"r_phi (max. bound): {r_phi}\nmax_u: {self.max_acceleration}\na: {a}\nn: {n}")
