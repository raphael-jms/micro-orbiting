import numpy as np
import sympy as sp
import time
import scipy.linalg as la
import scipy.signal
import math
import cvxpy as cp
import casadi as ca

from micro_orbiting_mpc.util.utils import read_yaml_matrix, EllipticalTerminalConstraint
from micro_orbiting_mpc.util.get_example import get_faulty_ff_spiral
from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved, SpiralParameters

class TerminalIncredients:
    def __init__(self, model, params):
        self.model = model
        self.params = params

        self.r = self.model.spiral_params.r
        self.omega_theta = self.model.spiral_params.omega_theta

        self.mass = self.model.mass
        self.J = self.model.J

        self.M = np.array([
            [1/self.mass, 0, -self.r/self.J],
            [0, 1/self.mass, 0],
            [0, 0, 1/self.J]
        ])
        self.Minv = np.array([[self.mass, 0, self.r*self.mass],
                         [0, self.mass, 0],
                         [0, 0, self.J]])
    def get_termimal_cost(self):
        """
        The result will never be exactly the same if I once calculate in discrete and once in continuous time!
        Here discrete, is (maybe?) the more accurate thing to do
        """
        self.Q = np.diag(self.params["Q"])
        self.R = np.diag(self.params["R"])

        self.Q_lin = np.diag(self.params["Q_linfb"])
        self.R_lin = np.diag(self.params["R_linfb"])

        # The feedback-linearized system
        A_lin_cont = np.array([
            [0, 0, 1, 0, 0], # d c1/dt
            [0, 0, 0, 1, 0], # d c2/dt
            [0, 0, 0, 0, 0], # d c3/dt
            [0, 0, 0, 0, 0], # d c4/dt
            [0, 0, 0, 0, 0]  # d omega/dt
        ])
        B_lin_cont = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        C_lin_cont = np.eye(5)
        D_lin_cont = np.zeros((5,3))

        # print(scipy.signal.cont2discrete((A_lin_cont, B_lin_cont, C_lin_cont, D_lin_cont), self.model.dt))
        # res = scipy.signal.cont2discrete((A_lin_cont, B_lin_cont, C_lin_cont, D_lin_cont), self.model.dt)
        # print(res)
        [self.A_lin, self.B_lin, self.C_lin, self.D_lin, dt] = scipy.signal.cont2discrete((A_lin_cont, B_lin_cont, C_lin_cont, D_lin_cont), self.model.dt)

        # Other parameters
        self.K_lin_fb, self.P_lin_fb = self.get_linearized_feedback(self.A_lin, self.B_lin, self.Q_lin, self.R_lin)
        self.k_omega = self.K_lin_fb[-1, -1].item()

        # calculate P_Q
        self.A_K = self.A_lin - self.B_lin @ self.K_lin_fb
        self.P_Q = la.solve_discrete_lyapunov(self.A_K, self.Q)

        # calculate abs(Q_u)
        self.abs_Q_u = np.linalg.norm(self.Minv.T @ self.R @ self.Minv, ord=2)

        # calculate P_K
        self.P_K = la.solve_discrete_lyapunov(self.A_K, self.K_lin_fb.T @ self.K_lin_fb)

        # calculate gains
        gains = [
            (2*self.r*self.omega_theta) ** 2 / (1 - (1-self.k_omega)**2),
            4*self.r*self.omega_theta / (1 - (1-self.k_omega)**3),
            self.r**2 / (1 - (1-self.k_omega)**4),
        ]
        self.gains = gains

        def cost_fcn(e0):
            e0 = e0[0:5] # strip potential alpha

            e0_5 = e0[4]
            return e0.T @ self.P_Q @ e0 + 2 * self.abs_Q_u * \
                (e0.T @ self.P_K @ e0 + gains[0]*e0_5**2 + gains[1]*e0_5**3 + gains[2]*e0_5**4)

        # return a function that can be called
        return cost_fcn

    def get_linearized_feedback(self, A, B, Qx, Qu):
        # P_lin_fb = la.solve_continuous_are(A, B, Qx, Qu)
        P_lin_fb = la.solve_discrete_are(A, B, Qx, Qu)

        K = la.inv(Qu) @ (B.T) @ P_lin_fb
        K[np.abs(K)<0.00001] = 0

        return K, P_lin_fb

    def calculate_terminal_set(self, trajectory=None):
        input_bounds = InputBounds(self.model)
        A, b = input_bounds.get_conv_hull() # A, b for complete force
        b = b.reshape(-1,1) - A @ self.model.faulty_input_simple # A, b, for control force

        upper_bound = cp.Variable()
        lam = cp.Variable()
        constraints = []

        def get_s_procedure_constraint(F1, g1, h1, F2, g2, h2, lam):
            """
            Formulate
            x^T F1 x + 2 g1 x + h1 <= 0    ==>    x^T F2 x + 2 g2 x + h2 <= 0
            as a definiteness constraint
            """
            dimX = max(F1.shape[1], F2.shape[1])
            F1 = np.zeros((dimX, dimX)) if F1 is None else F1
            F2 = np.zeros((dimX, dimX)) if F2 is None else F2
            g1 = np.zeros((dimX, 1)).reshape(-1,1) if g1 is None else g1
            g2 = np.zeros((dimX, 1)).reshape(-1,1) if g2 is None else g2

            mat1 = cp.vstack((
                cp.hstack((F1, g1)),
                cp.hstack((g1.T, cp.bmat([[h1]])))
            ))
            mat2 = cp.vstack((
                cp.hstack((F2, g2)),
                cp.hstack((g2.T, cp.bmat([[h2]])))
            ))
            return mat1 - lam * mat2 >> 0

        r = self.r
        omega_theta = self.omega_theta
        Minv = self.Minv

        # Build the constraint on feasibility of the control law; constructed from parts c1-c3
        # Calculate c1
        c1 = r

        # Calculate c2
        """
        Somehow this is easily doable in Maple, but not in Python. It turns however out that
        the c2 is always the same for different alpha values. Since I don't have a rigorous
        proof for this, I verify this by sampling alpha. The proof comes later (hopefully).
        """
        # TODO Proof
        c2s = []
        for alpha in np.linspace(0, 2*np.pi, 100):
            mat = np.zeros_like(self.K_lin_fb)
            mat[0,4] = -np.sin(alpha) * 2*r*omega_theta
            mat[1,4] =  np.cos(alpha) * 2*r*omega_theta
            c2s.append(np.linalg.norm(-self.K_lin_fb + mat))

        c2s = np.array(c2s)
        if not np.all(np.isclose(c2s, c2s[0])):
            raise ValueError("c2s are not equal for different alpha values.")
        c2 = c2s[0]

        # Calculate c3
        traj_2der = 0.1
        traj_2der = np.linalg.norm(b, 2) / np.linalg.norm(A @ Minv, 2) * 0.4
        c3 = np.linalg.norm(b, 2) / np.linalg.norm(A @ Minv, 2) - traj_2der

        ctemp = np.zeros((5,5)); ctemp[4,4] = 2*c1*c2
        C = np.eye(5) * c2**2 + ctemp

        """
        Now, the condition e^T C e <= c3**2 is ready to be fulfilled
        """
        constraints.append(get_s_procedure_constraint(self.P_lin_fb, None, -upper_bound,
                                                        C, None, -c3.item()**2, 
                                                        lam))

        # Solve the problem
        objective = cp.Maximize(upper_bound)
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.SCS)

        terminal_set = EllipticalTerminalConstraint(prob.value, self.P_lin_fb)
        # print(f"Terminal set: {terminal_set}")
        return terminal_set
    
if __name__ == "__main__":
    model = get_faulty_ff_spiral()
    term = TerminalIncredients(model)
    breakpoint()
