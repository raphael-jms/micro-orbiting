import numpy as np
import casadi as ca
import casadi.tools as ctools
import time
import matplotlib.pyplot as plt
import warnings
import time
from ament_index_python.packages import get_package_share_directory
import os

from micro_orbiting_msgs.msg import ControllerValues
from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved, SpiralParameters, PlottingHelper
from micro_orbiting_mpc.controllers.spiralMPC_linearizing.terminal_incredients_linearizing import TerminalIncredients
from micro_orbiting_mpc.controllers.controller_base_class import ControllerBaseClass
from micro_orbiting_mpc.controllers.controller_mpc_base import GenericMPC
from micro_orbiting_mpc.util.terminal_constraints import PolytopicTerminalConstraint
from micro_orbiting_mpc.util.utils import LogData, read_yaml_matrix, EllipticalTerminalConstraint, Rot, RotInv, Rot3, Rot3Inv


class SpiralMPC(GenericMPC):
    DEFAULT_PARAMS = GenericMPC.DEFAULT_PARAMS.copy()
    DEFAULT_PARAMS["trajectory_tracking"] = True
    DEFAULT_PARAMS["terminal_constraint"] = "set"
    DEFAULT_PARAMS["recalculate_terminal_set"] = False

    def __init__(self, model, params, ros_node, include_omega=True):
        self.include_omega = include_omega # Include omega in the optimization variables

        self.spiral_params = SpiralParameters(model)
        self.terminal_incredients = TerminalIncredients(model, params["tuning"][params["param_set"]])

        super().__init__(model, params, ros_node)

    def set_cost_functions(self):
        # For running cost, the code from the parent class is used
        super().set_cost_functions()
        self.terminal_cost = self.terminal_incredients.get_termimal_cost()

    def build_solver(self):
        """
        Build solver. All state variables are in spiral center form
        """
        build_solver_start = time.time()

        # Cost function weights
        Q = self.dict2matrix("Q")
        R = self.dict2matrix("R")

        self.Q = ca.MX(Q)
        self.R = ca.MX(R)

        self.x_sp = None
        self.u_sp = None

        x0 = ca.MX.sym('x0', self.Nx)

        if self.trajectory_tracking:
            # create vector representing the reference trajectory
            x_ref = ca.reshape(ca.MX.sym('x_ref', self.Nopt, (self.Nt+1)), (-1, 1))
            u_ref = ca.reshape(ca.MX.sym('u_ref', self.Nu, (self.Nt+1)), (-1, 1))
        else:
            # If no trajectory tracking: x_ref is the desired steady-state
            x_ref = ca.MX.sym('x_ref', self.Nopt,)
            u_ref = ca.MX.sym('u_ref', self.Nu,)

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

        # Bounds on x
        xub = self.get_param("xub")
        xlb = self.get_param("xlb")
        # Bounds on u
        ChullMat, ChullVec = self.bounds.get_conv_hull() # A, b

        # Compensating input to get the virtual incontrollable force, not the pysical one
        u_comp = self.spiral_params.compensation_force
        self.u_comp = u_comp
        u_uncontrolled = self.model.faulty_input_simple.flatten()

        # Generate MPC Problem
        for t in range(self.Nt):
            # Get variables
            x_t = opt_var['x', t]
            if self.trajectory_tracking:
                x_r = x_ref[t*self.Nopt:(t+1)*self.Nopt]
                u_r = u_ref[t*self.Nu:(t+1)*self.Nu]
                # The saved nominal input is not yet corrected for the orientation, correct now...
                alpha = x_t[5]
                rot = ca.MX(2,2)
                rot[0, 0] = ca.cos(alpha)
                rot[0, 1] = ca.sin(alpha)
                rot[1, 0] = -ca.sin(alpha)
                rot[1, 1] = ca.cos(alpha)
                u_r = ca.vcat([ca.mtimes(rot, u_r[0:2]), u_r[2]])
            else:
                x_r = x_ref
                u_r = u_ref
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t + u_r + u_comp) 
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            con_ineq.append(ChullMat @ (u_t + u_r + u_comp + u_uncontrolled))
            con_ineq_ub.append(ChullVec) # u_comp taken into account here
            # con_ineq.append(ChullMat @ (u_t + u_r))
            # con_ineq_ub.append(ChullVec - ChullMat @ (u_comp + u_uncontrolled)) # u_comp taken into account here
            # con_ineq.append(ChullMat @ (u_t + u_r + u_comp))
            # con_ineq_ub.append(ChullVec) # u_comp taken into account here
            con_ineq_lb.append(np.full(ChullVec.shape, -ca.inf))

            # State constraints
            if not(xub is None and xlb is None):
                xub = np.full((self.Nx,), ca.inf) if xub is None else xub
                xlb = np.full((self.Nx,), -ca.inf) if xlb is None else xlb
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t[0:self.Nopt], x_r, self.Q, u_t, self.R)

        # Terminal incredients
        x_t = opt_var['x', self.Nt]
        if self.trajectory_tracking:
            x_r = x_ref[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt]
        else:
            x_r = x_ref

        # Terminal Cost
        e_N = x_t[0:self.Nopt] - x_r

        # Terminal Constraint
        if self.get_param("terminal_constraint") == "set":
            obj += self.terminal_cost(e_N)

            terminal_constraint = self.terminal_incredients.calculate_terminal_set()
            self.terminal_constraint = terminal_constraint
            self.mpc_params["terminal_constraint"] = terminal_constraint
            P = terminal_constraint.P
            alpha = terminal_constraint.alpha
            con_ineq.append(ca.mtimes(ca.mtimes(e_N.T, P), e_N))
            con_ineq_lb.append(-ca.inf)
            # con_ineq_ub.append(alpha/3)
            con_ineq_ub.append(alpha)
        elif self.get_param("terminal_constraint") == "point":
            print("Using point_term_constraint")
            # Terminal constraint as single point
            con_eq.append(e_N)
        else:
            self.logger.error("Invalid terminal constraint")

        # Equality constraints are reformulated as inequality constraints with 0<=g(x)<=0
        # -> Refer to CasADi documentation: NLP solver only accepts inequality constraints
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
        # self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)
        self.solver = ca.nlpsol('spiral_MPC_sol', 'ipopt', nlp, options)

        print('\n________________________________________')
        print(f"# Time to build mpc solver: {time.time() - build_solver_start} sec")
        print(f"# Number of variables: {self.num_var}")
        print(f"# Number of equality constraints: {num_eq_con}")
        print(f"# Number of inequality constraints: {num_ineq_con}")
        print(f"# Horizon steps: {self.Nt * self.dt}s into the future")
        print('----------------------------------------')

    def get_control(self, x0, t, input_is_in_center=False):
        if not input_is_in_center:
            c0 = self.model.robot_to_center(x0)
        else:
            c0 = x0.copy()
        c0 = np.array(c0).flatten()

        if self.trajectory_tracking:
            x_ref, u_ref = self.get_next_trajectory_part(t)
            self.x_sp = x_ref.reshape(-1, 1, order='F')
            self.u_sp = u_ref.reshape(-1, 1, order='F')
        else:
            if self.x_sp is None or self.u_sp is None:
                self.x_sp = np.zeros(self.Nopt)
                self.u_sp = np.zeros(self.Nu)
        
        u_nom_alpha_corrected = (Rot3Inv(c0[5]) @ self.u_sp[0:self.Nu]).flatten()
        # Solve the optimization problem
        c, u, slv_time, cost, slv_status = self.solve_mpc(c0)
        u_res = np.array(u[0]).flatten() + u_nom_alpha_corrected + self.u_comp

        # # Overwrite MPC, just use nominal values
        # c, u, slv_time, cost, slv_status = [1], [[0, 0, 0]], 3, 4, 5
        # u_res = u_nom_alpha_corrected + self.u_comp

        if False:
            self.data["x"].add_data(t, x0)
            self.data["u"].add_data(t, u_res)
            self.data["u_nom"].add_data(t, u_nom_alpha_corrected)
            self.data["u_control"].add_data(t, u[0])
            self.data["e"].add_data(t, x0[0:self.Nopt] - self.x_sp[0:self.Nopt].flatten())
            self.data["slv_time"].add_data(t, slv_time)
            self.data['control_cost'].add_data(t, cost)
            self.data['solver_state'].add_data(t, slv_status)
            term_cost, running_x, running_u = self.calculate_cost_parts(c, u)
            self.data['term_cost'].add_data(t, term_cost)
            self.data['running_cost_x'].add_data(t, running_x)
            self.data['running_cost_u'].add_data(t, running_u)
            self.data['c'].add_data(t, c0)
            self.data['ce'].add_data(t, c0[0:self.Nopt] - self.x_sp[0:self.Nopt].flatten())

        u_phys = self.ih.get_physical_input(u_res)
        self.publish_last_controller_values(
            t,
            x0,
            self.x_sp,
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

        if self.plot_plannned_traj:
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color = colors[int(t/self.dt) % 5]
            c_planned = np.array(c).reshape(self.Nt+1,-1, order='F').T
            self.axs_planned_traj.plot(c_planned[0,:], c_planned[1,:], color)
            for i in range(6):
                self.axs_planned_states[i].plot([t+i*self.dt for i in range(self.Nt+1)], c_planned[i,:], color)
                self.axs_planned_states[i].plot(t, c_planned[i,0], color + "o")

        return u_phys 

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

        x_plan = x_plan.reshape(5, -1, order='F')
        msg.plan_x1 = [x[0].__float__() for x in x_plan]
        msg.plan_y1 = [x[1].__float__() for x in x_plan]
        msg.plan_alpha = [x[2].__float__() for x in x_plan]
        msg.plan_x2 = [x[3].__float__() for x in x_plan]
        msg.plan_y2 = [x[4].__float__() for x in x_plan]
        msg.plan_omega = [x[5].__float__() for x in x_plan]

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
        msg.center_state_vx = center[2].item()
        msg.center_state_vy = center[3].item()

        msg.center_error_x = center_error[0].item()
        msg.center_error_y = center_error[1].item()
        msg.center_error_omega = center_error[4].item()
        msg.center_error_vx = center_error[2].item()
        msg.center_error_vy = center_error[3].item()

        self.controller_stats_pub.publish(msg)

    def calculate_cost_parts(self, c, u):
        cc_final = c[self.Nt][0:self.Nopt]
        cc_ref_final = self.x_sp[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt].flatten()
        term_cost = self.terminal_cost(cc_final - cc_ref_final)

        running_u, running_x = 0, 0
        for i in range(self.Nt):
            uu = np.array(u[i]).reshape(-1,1)
            running_u += np.array(ca.evalf(uu.T @ self.R @ uu))
            cc = np.array(c[i]).reshape(-1,1)
            cc_ref = self.x_sp[i*self.Nopt:(i+1)*self.Nopt].reshape(-1,1)
            ee = cc[0:self.Nopt] - cc_ref
            running_x += np.array(ca.evalf(ee.T @ self.Q @ ee))

        return term_cost, running_x, running_u

    def prepare_logging(self):
        super().prepare_logging()
        self.data.update({
            # 'c': LogData(self.dt, ['c1', 'c2', 'c3', 'c4', 'omega', 'alpha']), 
            'c': LogData(self.dt, ['$c_1$', '$c_2$', '$c_3$', '$c_4$', '$\omega$', '$\\alpha$']), 
            'term_cost': LogData(self.dt, ['terminal_cost']),
            'running_cost_x': LogData(self.dt, ['running_cost_x']),
            'running_cost_u': LogData(self.dt, ['running_cost_u']),
            # include omega if Nopt=5
            'ce': LogData(self.dt, ['ce1', 'ce2', 'ce3', 'ce4', 'ce5'][0:self.Nopt])})

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

    def plot_internal_vals(self, filename=None, filepath="data/simulations/"):
        """
        Plots the internal values. If the filename is given, the plot is saved to the file.
        """
        # plot center point and error
        c, time = self.data["c"].get_array()
        ce, _ = self.data["ce"].get_array()

        if not self.trajectory_tracking:
            traj = np.zeros((self.Nopt, c.shape[1]))
        else:
            traj = self.trajectory[:, 0:c.shape[1]]

        fig, axs = plt.subplots(c.shape[0], 1, figsize=(10, 10))
        plt_titles_c = self.data["c"].name
        print(c.shape)
        for i in range(c.shape[0]):
            axs[i].step(time, c[i, :], where='post', label='center', color='blue')
            if traj.shape[0] > i:
                # alpha never has a reference
                axs[i].plot(time, traj[i, :], "--", label='reference', color='lightgrey')
                axs[i].step(time, ce[i, :], "--", where='post', label='error', color='red')
            axs[i].set_title(plt_titles_c[i])
            axs[i].legend()
            axs[i].grid()
        # fig.suptitle('Center and Error - Internal values')

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filepath + filename + "_states.png")

        # Plot input
        u, t = self.data["u"].get_array()
        u_nom, t = self.data["u_nom"].get_array()
        u_cont, t = self.data["u_control"].get_array()
        # u = u.reshape((self.Nu, -1)); u_nom = u_nom.reshape((self.Nu, -1)); u_cont = u_cont.reshape((self.Nu, -1))
        u_comp = np.tile(self.spiral_params.compensation_force.reshape(-1,1), (1, u.shape[1]))

        fig, axs = plt.subplots(3, 1)
        plt_titles = ['$F_x$', '$F_y$', '$T$']
        legends = ['$u_{full}$', '$u_{MPC}$', '$u_{nom}$', '$u_{compensate}$']
        colors = ['blue', 'red', 'black', 'grey']

        for i in range(3):
            axs[i].plot(t,  u[i, :], color=colors[0], label=legends[0])
            axs[i].plot(t,  u_cont[i, :], color=colors[1], label=legends[1])
            axs[i].plot(t,  u_nom[i, :], color=colors[2], label=legends[2])
            axs[i].plot(t,  u_comp[i, :], color=colors[3], label=legends[3])
            axs[i].set_title(plt_titles[i])
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filepath + filename + "_inputs.png")

        self.plot_control_cost(filepath + filename)

        # u_phys, u_nom_phys, u_cont_phys = [], [], []
        # u, t = self.data["u"].get_array()
        # u_nom, t = self.data["u_nom"].get_array()
        # u_cont, t = self.data["u_control"].get_array()

        # for i in range(u.shape[1]):
        #     u_phys.append(self.ih.get_physical_input(u[:, i]))
        #     u_nom_phys.append(self.ih.get_physical_input(u_nom[:, i]))
        #     u_cont_phys.append(self.ih.get_physical_input(u_cont[:, i]))
        
        # u_phys = np.array(u_phys).T
        # u_nom_phys = np.array(u_nom_phys).T
        # u_cont_phys = np.array(u_cont_phys).T

        # self.plot_physical_input(u_phys, u_nom_phys, u_cont_phys, t)

        if self.plot_plannned_traj:
            no_points = max(time.shape)
            self.axs_planned_traj.plot(self.trajectory[0,0:no_points], self.trajectory[1,0:no_points], 'k')
            self.axs_planned_traj.plot(c[0,0:no_points], c[1,0:no_points], '--', color='lightgrey')
            self.axs_planned_traj.set_title(f"Horizon: {self.Nt}\nBlack: trajectory; red: planned trajectory (horizon), grey: actual travelled path")

            for i in range(5):
                self.axs_planned_states[i].plot(t[0:no_points],self.trajectory[i,0:no_points], 'k')
                self.axs_planned_states[i].plot(t[0:no_points],c[i,0:no_points], '--', color='lightgrey')

            self.axs_planned_states[0].text(0.99, 0.99, "Spiral center position x", ha='right', va='top', transform=self.axs_planned_states[0].transAxes)
            self.axs_planned_states[1].text(0.99, 0.99, "Spiral center position y", ha='right', va='top', transform=self.axs_planned_states[1].transAxes)
            self.axs_planned_states[2].text(0.99, 0.99, "Spiral center velocity x", ha='right', va='top', transform=self.axs_planned_states[2].transAxes)
            self.axs_planned_states[3].text(0.99, 0.99, "Spiral center velocity y", ha='right', va='top', transform=self.axs_planned_states[3].transAxes)
            self.axs_planned_states[4].text(0.99, 0.99, "omega", ha='right', va='top', transform=self.axs_planned_states[4].transAxes)
            self.axs_planned_states[5].text(0.99, 0.99, "alpha", ha='right', va='top', transform=self.axs_planned_states[5].transAxes)
            self.fig_states.suptitle(f"Horizon: {self.Nt}")

    @property
    def Nopt(self):
        return 5 if self.include_omega else 4
