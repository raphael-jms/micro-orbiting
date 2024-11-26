import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
import warnings

import cvxpy as cp

from micro_orbiting_mpc.controllers.controller_base_class import ControllerBaseClass
from micro_orbiting_mpc.util.utils import Rot3, Rot3Inv, read_yaml_matrix, LogData
from micro_orbiting_mpc.util.polytope import MyPolytope

from micro_orbiting_mpc.controllers.spiralMPC_linearizing.terminal_incredients_linearizing import TerminalIncredients

class FBLinearizingController(ControllerBaseClass):
    def __init__(self, model):
        super().__init__()
        self.set_model(model)

        self.r = self.model.spiral_params.r
        self.omega_theta = self.model.spiral_params.omega_theta
        self.b = self.model.spiral_params.b
        self.beta = math.atan2(self.b[1], self.b[0])
        self.u_comp = self.model.spiral_params.compensation_force.reshape(-1, 1)

        print(f"r: {self.r}, omega_theta: {self.omega_theta}, b: {self.b.T}, u_comp: {self.u_comp.T}, beta: {self.beta}")

        # get polytope in the (global) input space
        A, b = self.ih.input_bounds.get_conv_hull()
        # polytope centered at the virtual uncontrollable force
        self.input_constr_centered_at_b = MyPolytope(A, b).minkowski_add_vector(-self.b)

        self.spiral_params = self.model.spiral_params

        self.trajectory_tracking=True

        self.Nt = 1

        A_lin = np.array([
            [0, 0, 1, 0, 0], # d c1/dt
            [0, 0, 0, 1, 0], # d c2/dt
            [0, 0, 0, 0, 0], # d c3/dt
            [0, 0, 0, 0, 0], # d c4/dt
            [0, 0, 0, 0, 0]  # d omega/dt
        ])
        B_lin = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        Q_lin = read_yaml_matrix("./controllers/tuning.yaml", "fb_lin_5", "P1", "Q_linfb")
        R_lin = read_yaml_matrix("./controllers/tuning.yaml", "fb_lin_5", "P1", "R_linfb")

        self.A_lin = A_lin; self.B_lin = B_lin; self.Q_lin = Q_lin; self.R_lin = R_lin

        self.K, self.P = self.get_linearized_feedback(A_lin, B_lin, Q_lin, R_lin)

        self.mass = model.mass
        self.J = model.J

        self.use_beta = False

        if self.use_beta:
            self.Minv = np.array([[self.mass, 0, -np.sin(self.beta)*self.r*self.mass],
                                [0, self.mass, np.cos(self.beta)*self.r*self.mass],
                                [0, 0, self.J]])
        else:
            self.Minv = np.array([[self.mass, 0, self.r*self.mass],
                                [0, self.mass, 0],
                                [0, 0, self.J]])

        self.data = {
            'c': LogData(self.dt, ['$c_1$', '$c_2$', '$c_3$', '$c_4$', '$\omega$', '$\\alpha$']), 
            # "c": LogData(model.dt, ["c1", "c2", "c3", "c4", "c5", "c6"]),
            "e": LogData(model.dt, ["e1", "e2", "e3", "e4", "e5", "e6"]),
            "f": LogData(model.dt, ["Fx", "Fy", "T"]),
            "f_nonconst": LogData(model.dt, ["Fx", "Fy", "T"]),
            "cost": LogData(model.dt, ["cost"]),
            "u": LogData(model.dt, ["$F_x$", "$F_y$", "$T$"])
            }

    def assign_trajectory(self, traj):
        self.trajectory = np.vstack((traj[0:2, :], traj[3:5, :], self.omega_theta*np.ones_like(traj[0, :])))
        x = self.trajectory[0:3, :]
        self.secondDer = np.gradient(np.gradient(x, axis=1), axis=1) / self.dt**2

    def get_control(self, x0, t, input_is_in_center=False):
        if not input_is_in_center:
            c0 = self.model.robot_to_center(x0)
        else:
            c0 = x0
        alpha = np.array(c0[5]).item()

        x_r, u_r = self.get_next_trajectory_part(t)
        u_r = u_r[:, 0].reshape(-1, 1)

        e = c0[0:5] - x_r[0:5, 0]
        e = np.array(e)
        e5 = e[4].item()

        f_e = e5 * self.r * (e5 + 2 * self.omega_theta)

        if self.use_beta:
            u = - np.array([[np.cos(alpha + self.beta), np.sin(alpha+self.beta), 0]]).T * f_e + u_r - self.K @ e
        else:
            u = Rot3(alpha) @ np.array([[0,f_e,0]]).T + u_r - self.K @ e

        F = self.Minv @ Rot3Inv(alpha) @ u
        F = self.clip_to_input_bounds(F.flatten())
        F += self.u_comp.flatten()

        # resulting_input = self.clip_to_input_bounds(F.flatten())

        # F = np.zeros((3,1))

        self.data["c"].add_data(t, c0)
        self.data["e"].add_data(t, e)
        self.data["f"].add_data(t, F.flatten())
        self.data["cost"].add_data(t, e.T @ self.P @ e)
        if self.use_beta:
            raise NotImplementedError("Not implemented")
        else:
            self.data["f_nonconst"].add_data(t, self.Minv @ Rot3Inv(alpha) @ (Rot3(alpha) @ np.array([[0,f_e,0]]).T - self.K @ e))

        # return resulting_input
        return self.ih.get_physical_input(F.flatten())
        # warnings.warn("Giving back the 3dim input")
        # return F.flatten()

    def get_next_trajectory_part(self, t):
        """
        Get the next points in the trajectory that lies within the prediction horizon.
        """
        id_s = int(round(t / self.dt))
        id_e = int(round(t / self.dt)) + self.Nt + 1
        x_r = self.trajectory[:, id_s:id_e]
        u_r = self.secondDer[:, id_s:id_e]
        return x_r, u_r

    def get_linearized_feedback(self, A, B, Qx=None, Qu=None):
        # penalize alpha a lot → staying within close reagion for alpha <=> bigger region for rest
        # Penalize control input a lot → Bigger resulting region with possible control input
        if Qx is None:
            Qx = np.diag([1,1,100,1,1,1])
        if Qu is None:
            Qu = np.diag([1,1,1]) * 10

        P_lin_fb = la.solve_continuous_are(A, B, Qx, Qu)
        K = la.inv(Qu) @ (B.T) @ P_lin_fb

        K[np.abs(K)<0.00001] = 0

        return K, P_lin_fb

    def clip_to_input_bounds(self, u):
        A = self.input_constr_centered_at_b.A
        b = self.input_constr_centered_at_b.b # ignore that technically b is used twice as virtual force and as constraint...

        if np.all(A @ u <= b):
            # constraints satisfied
            return u

        mult = cp.Variable()
        objective = cp.Maximize(mult)
        constraints = [A @ (mult * u) <= b, mult >= 0, mult <= 1]

        problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.ECOS) # SCS fails for some reason...
        problem.solve(solver=cp.OSQP) # SCS fails for some reason...

        return u * mult.value

    def plot_internal_vals_old(self, filename=None, filepath="data/simulations/"):
        c, t = self.data["c"].get_array()
        e, t = self.data["e"].get_array()
        c = c.T; e = e.T

        fig, axs = plt.subplots(5, 3, figsize=(10, 10))
        for i in range(5):
            axs[i,0].plot(t, c[:, i], label=f"c{i+1}")
            axs[i,0].plot(t, e[:, i], label=f"e{i+1}")
            axs[i,0].legend()
            axs[i,0].grid()
            axs[i,1].plot(t, e[:, i], label=f"e{i+1}")
            axs[i,1].legend()
            axs[i,1].grid()

        F, t = self.data["f"].get_array()
        for i in range(3):
            axs[i,2].plot(t, F[i,:], label=f"f{i}")
            axs[i,2].legend()
            axs[i,2].grid()

        cost, t = self.data["cost"].get_array()
        axs[4,2].plot(t, cost.flatten(), label="cost")
        axs[4,2].legend()
        axs[4,2].grid()

        # if filename is not None:
        #     plt.savefig(filepath + filename + "_inputs.png")

    def plot_internal_vals(self, filename=None, filepath="data/simulations/"):
        """
        Plots the internal values. If the filename is given, the plot is saved to the file.
        """
        # plot center point and error
        c, time = self.data["c"].get_array()
        ce, _ = self.data["e"].get_array()

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
        u, t = self.data["f"].get_array()
        # u_nom, t = self.data["u_nom"].get_array()
        # u_cont, t = self.data["u_control"].get_array()
        # # u = u.reshape((self.Nu, -1)); u_nom = u_nom.reshape((self.Nu, -1)); u_cont = u_cont.reshape((self.Nu, -1))
        # u_comp = np.tile(self.spiral_params.compensation_force.reshape(-1,1), (1, u.shape[1]))

        fig, axs = plt.subplots(3, 1)
        plt_titles = ['$F_x$', '$F_y$', '$T$']
        legends = ['$u_{full}$', '$u_{MPC}$', '$u_{nom}$', '$u_{compensate}$']
        colors = ['blue', 'red', 'black', 'grey']

        for i in range(3):
            axs[i].plot(t,  u[i, :], color=colors[0], label=legends[0])
            # axs[i].plot(t,  u_cont[i, :], color=colors[1], label=legends[1])
            # axs[i].plot(t,  u_nom[i, :], color=colors[2], label=legends[2])
            # axs[i].plot(t,  u_comp[i, :], color=colors[3], label=legends[3])
            axs[i].set_title(plt_titles[i])
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filepath + filename + "_inputs.png")

        # self.plot_control_cost(filepath + filename)

        # if self.plot_plannned_traj:
        if False:
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



    def plot_cost_estimate(self):
        term = TerminalIncredients(self.model)

        recalculate_term_cost = True
        compare_pred_trajectory = True
        compare_control_inputs = True

        if recalculate_term_cost:
            term.calculate_terminal_cost()
        cost = term.load_terminal_cost()

        Q_cost = read_yaml_matrix("./controllers/tuning.yaml", "spiraling_5", "P1", "Q")
        R_cost = read_yaml_matrix("./controllers/tuning.yaml", "spiraling_5", "P1", "R")

        # calculate the terminal cost
        n_steps = self.data["c"].len()
        actual_cost = np.zeros((n_steps, 1))
        stage_cost = np.zeros((n_steps, 1))
        stage_cost_e = np.zeros((n_steps, 1))
        stage_cost_u = np.zeros((n_steps, 1))

        c, t = self.data["c"].get_array()
        e, t = self.data["e"].get_array()
        u, t = self.data["f_nonconst"].get_array()

        for i in range(n_steps-1, -1, -1):
            # Calculate current cost
            # As in continuous time, there is an integration taking place, the discrete integration has to be performed
            stage_cost_e[i] = e[:,i].T @ Q_cost @ e[:,i] * self.dt
            stage_cost_u[i] = u[:,i].T @ R_cost @ u[:,i] * self.dt
            stage_cost[i] = stage_cost_e[i] + stage_cost_u[i]
            actual_cost[i] = stage_cost[i]
            # Add following cost
            if i != n_steps-1:
                actual_cost[i] += actual_cost[i+1]

        estimated_cost = np.zeros((n_steps, 1))
        for i in range(n_steps):
            estimated_cost[i] = cost(e[:,i])

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t, actual_cost, "g-", label="Actual cost")
        ax[0].plot(t, estimated_cost, "r-", label="Estimated cost")
        ax[1].plot(t, stage_cost, "b-", label="Stage cost")
        ax[0].grid()
        ax[1].grid()
        ax[0].legend()
        ax[1].legend()

        if recalculate_term_cost:
            evolution = []
            for tt in t:
                evolution.append(
                    (term.mex(tt) @ e[:,0]).flatten()
                )

            evolution = np.array(evolution).T

            if compare_pred_trajectory:
                fig, axs = plt.subplots(5, 1, figsize=(10, 10))
                for i in range(5):
                    axs[i].plot(t, evolution[i, :], label=f"prediction")
                    axs[i].plot(t, e[i,:], label=f"actual trajectory")
                    axs[i].plot(t, evolution[i, :] - e[i,:], label=f"diff")
                    axs[i].legend()
                    axs[i].grid()

            r = self.model.spiral_params.r
            omega_theta = self.model.spiral_params.omega_theta

            if compare_control_inputs:
                alphas = c[5, :]
                fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                lin_fb = - self.K @ evolution
                lin_fb = (self.Minv @ Rot3Inv(alphas[i])) @ lin_fb
                nonlin_fb = np.zeros_like(lin_fb)
                for i in range(nonlin_fb.shape[1]):
                    nonlin_fb[:,i] = (self.Minv @ Rot3Inv(alphas[i]) @ np.array([[-np.sin(alphas[i]), np.cos(alphas[i]), 0]]).T * evolution[4,i] * r * (evolution[4,i] + 2 * omega_theta)).flatten()
                for i in range(3):
                    axs[i].plot(t, lin_fb[i,:], label=f"lin_fb")
                    axs[i].plot(t, nonlin_fb[i,:], label=f"nonlin_fb")
                    axs[i].plot(t, lin_fb[i,:] + nonlin_fb[i,:], label=f"sum")
                    axs[i].plot(t, u[i,:], label=f"actual")
                    axs[i].legend()
                    
            # calculate the remaining state cost
            cost_x_pred = np.zeros((n_steps, 1))
            cost_u_quad_pred = np.zeros((n_steps, 1))
            cost_u_nonl_pred = np.zeros((n_steps, 1))
            for i in range(n_steps):
                cost_x_pred[i] = e[:,i].T @ term.cost_x_quadr(*e[:,i], r, omega_theta) @ e[:,i]
                cost_u_quad_pred[i] = e[:,i].T @ term.cost_u_quadr(*e[:,i], r, omega_theta) @ e[:,i]
                cost_u_nonl_pred[i] = term.cost_nonlin_u(*e[:,i], r, omega_theta) 
            factor = term.cost_Pu_fact(r, omega_theta)

            cum_actual_cost_x = np.flip(np.cumsum(np.flip(stage_cost_e)))
            cum_actual_cost_u = np.flip(np.cumsum(np.flip(stage_cost_u)))

            fig, ax = plt.subplots(2, 2)
            ax[0,0].plot(t, cost_x_pred, "g-", label="Predicted cost")
            ax[0,0].plot(t, cum_actual_cost_x, "r-", label="Actual cost")
            ax[0,0].set_title("State cost")
            ax[0,0].legend()

            ax[0,1].plot(t, 2 * factor * (cost_u_quad_pred + cost_u_nonl_pred), "g-", label="Predicted cost")
            ax[0,1].plot(t, cum_actual_cost_u, "r-", label="Actual cost")
            ax[0,1].set_title("Control cost")
            ax[0,1].legend()

            ax[1,0].plot(t, 2 * factor * cost_u_quad_pred, "k-", label="Predicted cost")
            ax[1,0].set_title("Control cost quadratic part")
            ax[1,0].legend()

            ax[1,1].plot(t, 2 * factor * cost_u_nonl_pred, "k-", label="Predicted cost")
            ax[1,1].set_title("Control cost nonlinear part")
            ax[1,1].legend()

            print(f"factor: {factor}")

        
        breakpoint()

        
