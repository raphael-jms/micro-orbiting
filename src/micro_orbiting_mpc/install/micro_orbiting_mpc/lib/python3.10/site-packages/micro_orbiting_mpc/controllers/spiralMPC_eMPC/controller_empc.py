import numpy as np
import casadi as ca
import casadi.tools as ctools
import time
import matplotlib.pyplot as plt
import time
import random
import pprint
import copy

from micro_orbiting_msgs.msg import ControllerValues

from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved, SpiralParameters, PlottingHelper
from micro_orbiting_mpc.controllers.controller_mpc_base import GenericMPC
from micro_orbiting_mpc.util.utils import LogData, Rot3Inv, Rot3
from micro_orbiting_mpc.util.cost_handler import CostHandler

RobotToCenterRot = Rot3Inv 
CenterToRobotRot = Rot3


class FancyMPC(GenericMPC):
    DEFAULT_PARAMS = GenericMPC.DEFAULT_PARAMS.copy()
    DEFAULT_PARAMS["trajectory_tracking"] = True
    DEFAULT_PARAMS["terminal_constraint"] = "from_file"

    def __init__(self, model, params, robot_params, ros_node, include_omega=False):
        self.spiral_params = SpiralParameters(model)
        self.robot_params = robot_params

        super().__init__(model, params, ros_node)

    def set_cost_functions(self):
        super().set_cost_functions()

    def build_solver(self):
        """
        Build solver. All state variables are in spiral center form
        """
        build_solver_start = time.time()

        # Cost function weights
        Q = np.diag(self.tuning[self.param_set]["Q"])
        R = np.diag(self.tuning[self.param_set]["R"])

        self.Q = ca.MX(Q)
        self.R = ca.MX(R)

        self.x_sp = None
        self.u_sp = None

        x0 = ca.MX.sym('x0', self.Nx)

        # create vector representing the reference trajectory
        x_ref = ca.reshape(ca.MX.sym('x_ref', self.Nopt, (self.Nt+1)), (-1, 1))
        u_ref = ca.reshape(ca.MX.sym('u_ref', self.Nu, (self.Nt+1)), (-1, 1))

        param_s = ca.vertcat(x0, x_ref, u_ref)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt+1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)    

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Bounds on u in rob_local system
        ChullMat, ChullVec = self.bounds.get_conv_hull() # A, b
        beta = self.spiral_params.beta # Calculate instead in force-aligned system
        ChullMat = ChullMat @ CenterToRobotRot(beta - np.pi/2)

        print("center to robot rot\n", CenterToRobotRot(beta - np.pi/2))
        print("Robot to center rot\n", RobotToCenterRot(beta - np.pi/2))

        # Compensating input to get the virtual incontrollable force, not the pysical one
        u_comp = RobotToCenterRot(beta - np.pi/2) @ self.spiral_params.compensation_force
        self.u_comp = u_comp
        u_uncontrolled = RobotToCenterRot(beta - np.pi/2) @ self.model.faulty_input_simple.flatten()

        print(f"u_comp: {u_comp}")
        print(f"u_uncontrolled: {u_uncontrolled}")

        print(f"beta: {self.spiral_params.beta}")
        print(f"r: {self.spiral_params.r}")

        # Generate MPC Problem
        for t in range(self.Nt):
            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]

            x_r = x_ref[t*self.Nopt:(t+1)*self.Nopt]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t + u_comp) 
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            con_ineq.append(ChullMat @ (u_t + u_comp + u_uncontrolled))
            con_ineq_ub.append(ChullVec)
            con_ineq_lb.append(np.full(ChullVec.shape, -ca.inf))

            # Objective Function / Cost Function
            obj += self.running_cost(x_t[0:self.Nopt], x_r, self.Q, u_t, self.R)

        # Terminal cost
        x_t = opt_var['x', self.Nt]
        x_r = x_ref[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt]

        e_N = x_t[0:self.Nopt] - x_r
        obj += ca.transpose(e_N) @ ca.DM(np.diag([100, 100, 100, 100, 1000])) @ e_N

        # Equality constraints are reformulated as inequality constraints with 0<=g(x)<=0 -> Refer to CasADi documentation: NLP solver only accepts inequality constraints
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = { # maybe use jit compiler?
            'ipopt.print_level': 0,
            'print_time': False,
            'verbose': False,
            'expand': True
        }
        solver_opts = self.get_param("solver_opts")
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.nlpsol('spiral_MPC_sol', 'ipopt', nlp, options)

        self.logger.info('\n________________________________________')
        self.logger.info(f"# Time to build mpc solver: {time.time() - build_solver_start} sec")
        self.logger.info(f"# Number of variables: {self.num_var}")
        self.logger.info(f"# Number of equality constraints: {num_eq_con}")
        self.logger.info(f"# Number of inequality constraints: {num_ineq_con}")
        self.logger.info(f"# Horizon steps: {self.Nt * self.dt}s into the future")
        self.logger.info('----------------------------------------')

    def get_control(self, x0, t):
        c0 = self.model.robot_to_center(x0)
        c0 = np.array(c0).flatten()

        x_ref, u_ref = self.get_next_trajectory_part(t)
        self.x_sp = x_ref.reshape(-1, 1, order='F')
        self.u_sp = u_ref.reshape(-1, 1, order='F')
        
        u_nom_alpha_corrected = (Rot3Inv(c0[5]) @ self.u_sp[0:self.Nu]).flatten()
        # Solve the optimization problem
        c, u, slv_time, cost, slv_status = self.solve_mpc(c0)
        u_res = np.array(u[0]).flatten() + u_nom_alpha_corrected + self.u_comp
       
        print(f"u_res: {u_res}")

        beta = self.spiral_params.beta
        u_phys = self.ih.get_physical_input(CenterToRobotRot(beta - np.pi/2) @ u_res)

        print(f"u_res: {u_res} \t u_phys: {u_phys}")

        # beta = self.spiral_params.beta
        # print(f"error {RobotToCenterRot(beta - np.pi/2) @ self.model.faulty_input_simple.flatten()}")
        # print(f"np.array(u[0]).flatten() \t {RobotToCenterRot(beta - np.pi/2) @ np.array(u[0]).flatten()} + \n u_nom_alpha_corrected \t {RobotToCenterRot(beta - np.pi/2) @ u_nom_alpha_corrected} + \n self.u_comp \t\t {self.u_comp} \n = u_res \t\t\t {RobotToCenterRot(beta - np.pi/2) @ u_res}")

        self.publish_last_controller_values(
            t,
            x0,
            c,
            u_res,
            u_nom_alpha_corrected,
            u[0],
            x0[0:self.Nopt].flatten() - self.x_sp[0:self.Nopt].flatten(),
            slv_time,
            cost,
            slv_status,
            c0,
            c0[0:self.Nopt] - self.x_sp[0:self.Nopt].flatten(),
            u_phys
        )

        print(f"u_res: {u_res} \t u_phys: {u_phys}")

        beta = self.spiral_params.beta
        u_res = CenterToRobotRot(beta - np.pi/2) @ u_res
        u_phys = self.ih.get_physical_input(u_res)

        print(f"u_res 2: {u_res} \t u_phys: {u_phys}")

        return u_phys

    def calculate_cost_parts(self, c, u):
        cc_final = c[self.Nt][0:self.Nopt]
        cc_ref_final = self.x_sp[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt].flatten()
        term_cost = self.terminal_cost( *ca.vertsplit(cc_final - cc_ref_final))

        running_u, running_x = 0, 0
        for i in range(self.Nt):
            uu = np.array(u[i]).reshape(-1,1)
            running_u += np.array(ca.evalf(uu.T @ self.R @ uu))
            cc = np.array(c[i]).reshape(-1,1)
            cc_ref = self.x_sp[i*self.Nopt:(i+1)*self.Nopt].reshape(-1,1)
            ee = cc[0:self.Nopt] - cc_ref
            running_x += np.array(ca.evalf(ee.T @ self.Q @ ee))

        return term_cost, running_x, running_u

    def assign_trajectory(self, traj):
        # Prolong the trajectory to prevent the controller from running out of points 
        original_traj = np.hstack((traj, np.tile(traj[:, -1:], (1, self.Nt))))
        # Original traj in the form [x1, y1, alpha, x2, y2, omega]
        # New traj in the form [c1, c2, c3, c4, omega]
        self.trajectory = np.vstack((original_traj[0:2, :], original_traj[3:5, :],
                                     self.spiral_params.omega_theta * np.ones_like(original_traj[0, :])))
        self.trajectory = self.trajectory[0:self.Nopt, :]

        # Calculate the nominal input that realizes this trajectory
        # In this controller, this is not compensated for the angle as the angle is arbitraty
        # This has to be taken into account later on!
        x = self.trajectory[0:2, :]
        secondDer = np.gradient(np.gradient(x, axis=1), axis=1) / self.dt**2
        # include the mass (not inerita because omega_dot=0)
        necessary_force = np.vstack((secondDer[0:2, :] * self.mass, np.zeros_like(secondDer[0, :])))
        # necessary_force = -np.vstack((secondDer[0:2, :] * self.mass, np.zeros_like(secondDer[0, :])))
        self.nominal_input = necessary_force
    
    def publish_last_controller_values(self, t, x0, x_plan, u, u_nom, u_contr, e, slv_time, cost, 
                                       slv_status, center, center_error, u_phys):
        """
        Publish the last controller values to the ROS topic.
        Use only the values of ControllerValues that are relevant
        """
        msg = ControllerValues()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.header.frame_id = "controller_values"
        msg.x1 = x0[0].item()
        msg.y1 = x0[1].item()
        msg.alpha = x0[2].item()
        msg.x2 = x0[3].item()
        msg.y2 = x0[4].item()
        msg.omega = x0[5].item()

        msg.u = u.flatten().tolist()
        msg.u_nom = u_nom.flatten().tolist()
        msg.u_control = u_contr.full().flatten().tolist()
        msg.u_full = u_phys.flatten().tolist()

        # x_plan = x_plan.reshape(5, -1, order='F')
        x_plan = np.vstack([dm.full().flatten() for dm in x_plan]).T
        msg.plan_x1 = x_plan[0,:].tolist()
        msg.plan_y1 = x_plan[1,:].tolist()
        msg.plan_x2 = x_plan[2,:].tolist()
        msg.plan_y2 = x_plan[3,:].tolist()
        msg.plan_omega = x_plan[4,:].tolist()

        msg.e1 = e[0].item()
        msg.e2 = e[1].item()
        msg.e3 = e[2].item()
        msg.e5 = e[3].item()
        msg.e_omega = e[4].item()

        msg.solver_time = slv_time
        msg.control_cost = cost
        msg.solver_state = self.SOLVE_STATUS[slv_status]

        msg.center_state_x = center[0].item()
        msg.center_state_y = center[1].item()
        msg.center_state_omega = center[4].item()
        msg.center_state_alpha = center[5].item()
        msg.center_state_vx = center[2].item()
        msg.center_state_vy = center[3].item()

        msg.center_error_x = center_error[0].item()
        msg.center_error_y = center_error[1].item()
        msg.center_error_omega = center_error[4].item()
        msg.center_error_vx = center_error[2].item()
        msg.center_error_vy = center_error[3].item()

        self.controller_stats_pub.publish(msg)

    @property
    def Nopt(self):
        return 5 # include_omega always True
