import numpy as np
import casadi as ca
import casadi.tools as ctools
import time
import matplotlib.pyplot as plt
import warnings
import time

from micro_orbiting_msgs.msg import ControllerValues
from micro_orbiting_mpc.controllers.controller_base_class import ControllerBaseClass
from micro_orbiting_mpc.controllers.fb_linearizing_controller import FBLinearizingController
from micro_orbiting_mpc.util.terminal_constraints import PolytopicTerminalConstraint
from micro_orbiting_mpc.util.utils import LogData, read_yaml_matrix, EllipticalTerminalConstraint, Rot, RotInv, Rot3, Rot3Inv

class GenericMPC(ControllerBaseClass):
    # Default parameters for the MPC controller and its child classes
    DEFAULT_PARAMS = {
        "horizon": 10,
        "ulb": None,
        "uub": None,
        "xlb": None,
        "xub": None,
        "terminal_constraint": None,
        "param_set": "P1",
        "tuning": {},
        "solver_opts": None}

    # Encoding to send the status of the solver through ROS messages
    SOLVE_STATUS = {
        "Solve_Succeeded": 0,
        "Solved_To_Acceptable_Level": 1,
        "Infeasible_Problem_Detected": 2,
        "Search_Direction_Becomes_Too_Small": 3,
        "Diverging_Iterates": 4,
        "User_Requested_Stop": 5,
        "Maximum_Iterations_Exceeded": 6,
        "Restoration_Failed": 7,
        "Error_In_Step_Computation": 8,
        "Not_Enough_Degrees_Of_Freedom": 9,
        "Invalid_Problem_Definition": 10,
        "Invalid_Option": 11,
        "Invalid_Number_Detected": 12,
        "Unrecoverable_Exception": 13,
        "NonIpopt_Exception_Thrown": 14,
        "Insufficient_Memory": 15,
        "Internal_Error": 16,
        "Maximum_CpuTime_Exceeded": 17,
        "Feasible_Point_Found": 18
    }

    def get_param(self, param):
        """ Get the parameter value if set, default to DEFAULT_PARAMS if not """
        try:
            return self.mpc_params.get(param, self.DEFAULT_PARAMS[param])
        except KeyError:
            raise KeyError(f"Parameter '{param}' not found in mpc_params or DEFAULT_PARAMS")

    def __init__(self, model, params, ros_node):
        super().__init__(ros_node)
        self.mpc_params = params

        # set models, dynamics, bounds, input handler, model parameters
        self.set_model(model)

        self.trajectory = None

        self.tuning= self.get_param("tuning")
        self.param_set = self.get_param("param_set")

        self.Nt = self.get_param("horizon")
        # Number of optimized states
        self._Nopt = self.Nx
        self.prepare_logging()

        # Initialize variables
        self.set_cost_functions()

        self.build_solver()
        self.optimal_solution = None

        # self.plot_plannned_traj = True
        self.plot_plannned_traj = False
        if self.plot_plannned_traj:
            self.fig, self.axs_planned_traj = plt.subplots(1,1)
            self.fig_states, self.axs_planned_states = plt.subplots(6,1)

    def publish_last_controller_values(self, t, x0, x_plan, u, u_nom, u_contr, e, slv_time, cost, slv_status):
        """
        Publish the last controller values to the ROS topic.
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

        msg.u = u.full().flatten().tolist()
        msg.u_nom = u_nom.flatten().tolist()
        msg.u_control = u_contr.full().flatten().tolist()

        msg.plan_x1 = [x[0].__float__() for x in x_plan]
        msg.plan_y1 = [x[1].__float__() for x in x_plan]
        msg.plan_alpha = [x[2].__float__() for x in x_plan]
        msg.plan_x2 = [x[3].__float__() for x in x_plan]
        msg.plan_y2 = [x[4].__float__() for x in x_plan]
        msg.plan_omega = [x[5].__float__() for x in x_plan]

        msg.e1 = e[0].item()
        msg.e2 = e[1].item()
        msg.e_alpha = e[2].item()
        msg.e3 = e[3].item()
        msg.e5 = e[4].item()
        msg.e_omega = e[5].item()

        msg.center_state_x = x0[0].item()
        msg.center_state_y = x0[1].item()
        msg.center_state_omega = x0[5].item()
        msg.center_state_vx = x0[3].item()
        msg.center_state_vy = x0[4].item()

        msg.solver_time = slv_time
        msg.control_cost = cost
        msg.solver_state = self.SOLVE_STATUS[slv_status]

        self.controller_stats_pub.publish(msg)

    def dict2matrix(self, values):
        return np.diag(self.tuning[self.param_set][values])

    def build_solver(self):
        build_solver_start = time.time()

        # Cost function weights
        R_var = 'R' if self.Nu == 3 else 'R_full'
        Q = self.dict2matrix("Q")
        R = self.dict2matrix(R_var)
        P = self.tuning[self.param_set]["P_mult"] * Q

        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
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
        uub = self.get_param("uub")
        ulb = self.get_param("ulb")
        xub = self.get_param("xub")
        xlb = self.get_param("xlb")

        # Generate MPC Problem
        for t in range(self.Nt):
            # Get variables
            x_t = opt_var['x', t]
            x_r = x_ref[t*self.Nopt:(t+1)*self.Nopt]
            u_r = u_ref[t*self.Nu:(t+1)*self.Nu]
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t + u_r)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            if not(uub is None and ulb is None):
                uub = np.full((self.Nu,), ca.inf) if uub is None else uub
                ulb = np.full((self.Nu,), -ca.inf) if ulb is None else ulb
                con_ineq.append(u_t + u_r)
                con_ineq_ub.append(uub)
                con_ineq_lb.append(ulb)

            # State constraints
            if not(xub is None and xlb is None):
                xub = np.full((self.Nx,), ca.inf) if xub is None else xub
                xlb = np.full((self.Nx,), -ca.inf) if xlb is None else xlb
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(xlb)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t, x_r, self.Q, u_t, self.R)

        # Terminal incredients
        x_t = opt_var['x', self.Nt]
        x_r = x_ref[self.Nt*self.Nopt:(self.Nt+1)*self.Nopt]

        terminal_constraint = self.get_param("terminal_constraint")
        if isinstance(terminal_constraint, EllipticalTerminalConstraint):
            # Terminal constraint
            self.logger.info("Setting elliptical terminal constraint")
            self.logger.info(np.array2string(terminal_constraint.P))
            self.logger.info(str(terminal_constraint.alpha))
            P = terminal_constraint.P
            alpha = terminal_constraint.alpha
            con_ineq.append((x_t-x_r).T @ P @ (x_t-x_r))
            con_ineq_lb.append(-ca.inf)
            con_ineq_ub.append(alpha)

            # Terminal cost
            # ATTENTION In this case, term_constraint.P is used, NOT self.P!
            obj += self.terminal_cost(x_t, x_r, P)
        elif isinstance(terminal_constraint, PolytopicTerminalConstraint):
            # Terminal constraint
            # Should be a polytope
            H_N = terminal_constraint.A
            if H_N.shape[1] != self.Nx:
                raise ValueError("Terminal constraint with invalid dimensions.")

            H_b = terminal_constraint.b
            con_ineq.append(ca.mtimes(H_N, opt_var['x', self.Nt]))
            con_ineq_lb.append(-ca.inf * ca.DM.ones(H_N.shape[0], 1))
            con_ineq_ub.append(H_b)
            
            # Terminal cost
            obj += self.terminal_cost(x_t, x_r, self.P)
        elif terminal_constraint is None:
            self.logger.info("No terminal constraint set.")
        else:
            raise ValueError(f"Invalid terminal constraint: {terminal_constraint}")

        # Equality constraints are reformulated as inequality constraints with 0<=g(x)<=0
        # -> Refer to CasADi documentation: NLP solver only accepts inequality constraints
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        print(con_eq_lb)
        print(*con_ineq_lb)
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
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)

        self.logger.info('\n________________________________________')
        self.logger.info(f"# Time to build mpc solver: {time.time() - build_solver_start} sec")
        self.logger.info(f"# Number of variables: {self.num_var}")
        self.logger.info(f"# Number of equality constraints: {num_eq_con}")
        self.logger.info(f"# Number of inequality constraints: {num_ineq_con}")
        self.logger.info(f"# Horizon steps: {self.Nt * self.dt}s into the future")
        self.logger.info('----------------------------------------')

    def prepare_logging(self):
        self.data = {'x': LogData(self.dt, ['x1', 'y1', 'alpha', 'x2', 'y2', 'omega']),
                     'e': LogData(self.dt, ['e1', 'e2', 'e_alpha', 'e3', 'e5', 'e_omega']),
                     'u': LogData(self.dt, ['u11', 'u12', 'u21', 'u22', 'u31', 'u32', 'u41', 'u42']),
                     'u_nom': LogData(self.dt, ['u11_nom', 'u12_nom', 'u21_nom', 'u22_nom', 'u31_nom', 'u32_nom', 'u41_nom', 'u42_nom']),
                     'u_control': LogData(self.dt, ['u11_control', 'u12_control', 'u21_control', 'u22_control', 'u31_control', 'u32_control', 'u41_control', 'u42_control']),
                     'control_cost': LogData(self.dt, ['control_cost']),
                     'slv_time': LogData(self.dt, ['solver_time']),
                     'solver_state': LogData(self.dt, ['solver_state'])}

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nopt, self.Nopt)
        P = ca.MX.sym('P', self.Nopt, self.Nopt)
        R = ca.MX.sym('R', self.Nu, self.Nu)

        x = ca.MX.sym('x', self.Nopt)
        xr = ca.MX.sym('xr', self.Nopt)
        u = ca.MX.sym('u', self.Nu)

        # Prepare variables
        e_vec = x - xr

        # Calculate running cost
        ln = ca.mtimes(ca.mtimes(e_vec.T, Q), e_vec) + ca.mtimes(ca.mtimes(u.T, R), u)

        self.running_cost = ca.Function('ln', [x, xr, Q, u, R], [ln])

        # Calculate terminal cost
        V = ca.mtimes(ca.mtimes(e_vec.T, P), e_vec)
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

    def get_control(self, x0, t):
        x_ref, u_ref = self.get_next_trajectory_part(t)
        x0 = self.handle_angle_wraparound(x0, x_ref[:,0])
        self.x_sp = x_ref.reshape(-1, 1, order='F')
        self.u_sp = u_ref.reshape(-1, 1, order='F')

        # Solve the optimization problem
        x, u, slv_time, cost, slv_status = self.solve_mpc(x0)
        u_res = u[0] + self.u_sp[0:self.Nu]
        # u_res = self.u_sp[0:self.Nu]

        # u, slv_time = self.solve_mpc(x0, t)
        self.data["x"].add_data(t, x0)
        self.data["u"].add_data(t, u_res)
        self.data["u_nom"].add_data(t, self.u_sp[0:self.Nu])
        self.data["u_control"].add_data(t, u[0])
        self.data["e"].add_data(t, x0[0:self.Nopt].flatten() - self.x_sp[0:self.Nopt].flatten())
        self.data["slv_time"].add_data(t, slv_time)
        self.data['control_cost'].add_data(t, cost)
        self.data['solver_state'].add_data(t, slv_status)
        # print(f"t: {t}, x0: {x0.T}, x_sp: {self.x_sp[0:self.Nopt].T}, e: {(x0.flatten() - self.x_sp[0:self.Nopt].flatten()).T}, u: {u.T}")

        # Publish the controller values to ROS
        self.publish_last_controller_values(t, x0, x, u_res, self.u_sp[0:self.Nu], u[0], 
                                            x0[0:self.Nopt].flatten() - self.x_sp[0:self.Nopt].flatten(), 
                                            slv_time, cost, slv_status)
        return u_res

    def solve_mpc(self, x0):
        solver_start = time.time()
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # initialize variables
        # Initial guess of the warm start variables
        if self.optimal_solution is not None:
            self.optvar_init['x'] = self.optimal_solution['x'][1:] + [ca.DM([0] * self.Nx)]
            self.optvar_init['u'] = self.optimal_solution['u'][1:] + [ca.DM([0] * self.Nu)]
        else:
            self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        param = ca.vertcat(x0, self.x_sp, self.u_sp)
        
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])
        self.optimal_solution = optvar

        slv_time = time.time() - solver_start

        terminal = self.get_param("terminal_constraint")
        if isinstance(terminal, EllipticalTerminalConstraint):
            x = np.asarray(optvar['x', self.Nt])[0:self.Nopt]
            x_r = self.x_sp[self.Nt * self.Nopt : (self.Nt+1) * self.Nopt]
            self.logger.info(f"MPC - CPU time: {slv_time:,.7f} seconds  |  Cost: {float(sol['f']):9.2f}  |  Horizon length: {self.Nt}  |  Term. constraint: {((x-x_r).T @ terminal.P @ (x-x_r)).item():5.2f} <= {terminal.alpha:5.2f}  |  {status}")
        else:
            self.logger.info(f"MPC - CPU time: {slv_time:,.7f} seconds  |  Cost: {float(sol['f']):9.2f}  |  Horizon length: {self.Nt}  |  {status}")

        return optvar['x'], optvar['u'], slv_time, float(sol['f']), status

    def get_next_trajectory_part(self, t):
        """
        Get the next points in the trajectory that lies within the prediction horizon.
        """
        id_s = int(round(t / self.dt))
        id_e = int(round(t / self.dt)) + self.Nt + 1
        x_r = self.trajectory[:, id_s:id_e]
        u_r = self.nominal_input[:, id_s:id_e]

        print(f"t: {t}, id_s: {id_s}, id_e: {id_e}")
        print(f"x_r: {x_r[:,0].flatten()}")
        return x_r, u_r

    def handle_angle_wraparound(self, x0, x_r):
        """
        Angles are given in [-pi, pi]. This function maps them to [alpha_des - pi, alpha_des + pi]
        """
        # alpha_des = x_r[2]
        x0[2] = x0[2] + 2*np.pi * np.round((x_r[2] - x0[2]) / (2*np.pi))
        return x0

    def assign_trajectory(self, traj):
        # Prolong the trajectory to prevent the controller from running out of points 
        self.trajectory = np.hstack((traj, np.tile(traj[:,-1].reshape(-1,1), (1,self.Nt))))

        # Calculate the nominal input that realizes this trajectory
        x = self.trajectory[0:3, :]
        secondDer = np.gradient(np.gradient(x, axis=1), axis=1) / self.dt**2
        # include the mass and inertia
        necessary_force = np.vstack((secondDer[0:2, :] * self.mass, secondDer[2, :] * self.J))
        # Correct for the angle
        for i in range(necessary_force.shape[1]):
            # rot = np.array([[np.cos(x[2, i]), np.sin(x[2, i])],
            #                 [-np.sin(x[2, i]), np.cos(x[2, i])]])
            # necessary_force[0:2, i] = np.dot(rot, necessary_force[0:2, i])
            necessary_force[0:2, i] = RotInv(x[2, i]) @ necessary_force[0:2, i]
        # Convert the necessary simplified input to the full input
        self.nominal_input = np.empty((self.Nu, necessary_force.shape[1]))
        for i in range(necessary_force.shape[1]):
            self.nominal_input[:, i] = self.ih.get_physical_input(necessary_force[:, i])

    def plot_internal_vals(self):
        if self.data is {}:
            warnings.warn("No data to plot.")
            return

        traj = self.trajectory[:, 0:self.data["x"].len()]
        
        # plot state and error
        x, time = self.data["x"].get_array()
        e, _ = self.data["e"].get_array()

        fig, axs = plt.subplots(x.shape[0], 1, figsize=(10, 10))
        plt_titles_x = self.data["x"].name
        for i in range(x.shape[0]):
            axs[i].step(time, x[i, :], where='post', label='state', color='blue')
            axs[i].plot(time, traj[i, :], "--", label='reference', color='lightgrey')
            axs[i].step(time, e[i, :], "--", where='post', label='error', color='red')
            axs[i].set_title(plt_titles_x[i])
            axs[i].legend()
            axs[i].grid()
        # fig.suptitle('State and Error - Internal values')

        # plot input
        u, t = self.data["u"].get_array()
        u_nom, t = self.data["u_nom"].get_array()
        u_cont, t = self.data["u_control"].get_array()
        u = u.reshape((self.Nu, -1)); u_nom = u_nom.reshape((self.Nu, -1)); u_cont = u_cont.reshape((self.Nu, -1))
        self.plot_physical_input(u, u_nom, u_cont, t)

        self.plot_control_cost()
        plt.show(block=False)

    def plot_physical_input(self, u, u_nom, u_cont, t):
        fig, axs = plt.subplots(4, 1)
        plt_titles = ['F_1', 'F_2', 'F_3', 'F_4']
        legends = ['u+', 'u-', 'uc+', 'uc-', 'un+', 'un-']
        colors = ['blue', 'red', 'cyan', 'coral', 'lightsteelblue', 'orange']

        for i in range(4):
            axs[i].step(t,  u[2*i, :], color=colors[0], label=legends[0])
            axs[i].step(t, -u[2*i+1, :], color=colors[1], label=legends[1])
            axs[i].step(t,  u_cont[2*i, :], "--", color=colors[2], label=legends[2])
            axs[i].step(t, -u_cont[2*i+1, :], "--", color=colors[3], label=legends[3])
            axs[i].step(t,  u_nom[2*i, :], ":", color=colors[4], label=legends[4])
            axs[i].step(t, -u_nom[2*i+1, :], ":", color=colors[5], label=legends[5])
            axs[i].set_title(plt_titles[i])
            axs[i].grid(True)
            axs[i].legend()
            axs[i].set_ylim([-self.model.max_force*1.1, 1.1*self.model.max_force])
        

    def plot_control_cost(self, filename=None):
        cost, t = self.data["control_cost"].get_array()
        state, t = self.data["solver_state"].get_array()
        fig, ax = plt.subplots()
        # ax.plot(t, cost)

        current_state = state[0]
        start_index = 0
        for i in range(1, len(state)):
            if state[i] != current_state or i == len(state)-1:
                if current_state == 'Infeasible_Problem_Detected':
                    color = 'red'; label = 'Infeasible Problem'
                elif current_state == 'Solve_Succeeded':
                    color = 'green'; label = 'Problem solved'
                else:
                    color = 'blue'; label = 'Unknown state'
                ax.plot(t[start_index:i+1], cost[start_index:i+1], color=color, label=label)
                start_index = i
                current_state = state[i]

        if "term_cost" in self.data:
            t_cost, t = self.data["term_cost"].get_array()
            running_x, _ = self.data["running_cost_x"].get_array()
            running_u, _ = self.data["running_cost_u"].get_array()
            ax.plot(t.flatten(), t_cost.flatten(), label="Terminal Cost")
            ax.plot(t.flatten(), running_x.flatten(), label="Running Cost x")
            ax.plot(t.flatten(), running_u.flatten(), label="Running Cost u")
 
        ax.set_title("Control Cost")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.grid()

        if filename is not None:
            plt.savefig(filename + "_control_cost.png")

    @property
    def Nopt(self):
        return self._Nopt

class MultipleSamplingTimes:
    """
    Wrapper/Child class that enables the use of a different (lower) sampling time for the 
    controller than for the environment. 
    To the outside, the class imitates the behaviour of the normal controller.
    """
    def __init__(self, single_dt_controller_type, model, mpc_params, **kwargs):
        """
        Decorator class that gives controllers the ability to handle multiple sampling times.
        From the outside, thanks to __getattribute__, it looks and behaves like the 
        single_dt_controller.
        """
        if single_dt_controller_type is FBLinearizingController:
            self._controller = single_dt_controller_type(model, **kwargs)
        else:
            self._controller = single_dt_controller_type(model, mpc_params, **kwargs)
        # Default: Environment and controller have the same dt
        self.dt_env = mpc_params.get("environment_dt", self.dt)

        self.last_control = None

    def get_control(self, x0, t):
        """
        Only make a new calculation if the sim time is a multiple of the controller time.
        """
        if round(t/self.dt_env)%round(self.dt/self.dt_env) < 0.5:
            self.last_control = self._controller.get_control(x0, t)
        return self.last_control

    def __getattribute__(self, attr):
        if attr in ['get_control', 'dt_env', '_controller', 'last_control']:
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, '_controller'), attr)

    def __repr__(self) -> str:
        return repr(self._controller)