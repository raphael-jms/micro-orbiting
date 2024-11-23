import numpy as np
import sympy as sp
import time
import scipy.linalg as la
import math
import cvxpy as cp
import casadi as ca

from controllers.spiralMPC_linearizing.parse_sympy import python_code_from_sympy
from util.utils import read_yaml_matrix, EllipticalTerminalConstraint
from util.get_example import get_faulty_ff_spiral
from controllers.cost_function import get_cost_function
from models.ff_input_bounds import InputBounds, InputHandlerImproved, SpiralParameters

class TerminalIncredients:
    def __init__(self, model, tuning_file="./controllers/tuning.yaml", failure_case="spiraling_5", 
                 param_set="P1"):
        """
        Calculate the terminal incredients (terminal cost and constraints) for the feedback-
        linearized system.

        ATTENTION: state variables are in order
        state: [c1, c2, alpha c3, c4, omega]
            c1, c2 are the positions of the orbit center in x and y in global carthesian coords
            c3, c4 are the corresponding velocities
        This is due to (stupid) constraints of sympy. BUT: The load_terminal_cost function will
        return the correct order of [c1, c2, c3, c4, omega, alpha].
        """
        self.model = model
        # The feedback-linearized system
        self.A_lin = np.array([
            [0, 0, 1, 0, 0], # d c1/dt
            [0, 0, 0, 1, 0], # d c2/dt
            [0, 0, 0, 0, 0], # d c3/dt
            [0, 0, 0, 0, 0], # d c4/dt
            [0, 0, 0, 0, 0]  # d omega/dt
        ])
        self.B_lin = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Parametrized spiral parameters
        self.r = sp.symbols('r', real=True, positive=True)
        self.omega_theta = sp.symbols('omega_theta', real=True)

        # Other parameters
        self.mass = model.mass
        self.J = model.J

        self.M = np.array([
            [1/self.mass, 0, -self.r/self.J],
            [0, 1/self.mass, 0],
            [0, 0, 1/self.J]
        ])
        self.Minv = np.array([[self.mass, 0, self.r*self.mass],
                              [0, self.mass, 0],
                              [0, 0, self.J]])

        # The cost matrices for the function used in the MPC; Used to calculate terminal cost
        self.Q = read_yaml_matrix(tuning_file, failure_case, param_set, "Q")
        self.R = read_yaml_matrix(tuning_file, failure_case, param_set, "R")

        print(f"Q = {self.Q}")
        print(f"R = {self.R}")

        # Used to derive a controller for the terminal region
        Q_lin = read_yaml_matrix("./controllers/tuning.yaml", "spiraling_5", "P1", "Q_linfb")
        R_lin = read_yaml_matrix("./controllers/tuning.yaml", "spiraling_5", "P1", "R_linfb")
        self.K, self.P = self.get_linearized_feedback(self.A_lin, self.B_lin, Q_lin, R_lin)

        self.A_lin_cl = self.A_lin - self.B_lin @ self.K

    def calculate_terminal_cost(self):
        calculation_time = time.time()
        print("Calculating terminal cost...")

        r = self.r
        omega_theta = self.omega_theta

        t = sp.symbols('t', real=True)
        A_cl = sp.Matrix(self.A_lin_cl)
        mex = sp.simplify(sp.exp(A_cl * t)) # Matrix exponential, x(t) = mex * x(0)
        self.mex = sp.lambdify((t), mex)
        print(f"Calculation time matrix exponential: {time.time() - calculation_time} seconds")
        
        # Parametrized x(0)
        c0_1, c0_2, c0_3, c0_4, c0_5 = sp.symbols('c0_1 c0_2 c0_3 c0_4 c0_5')
        x0 = sp.Matrix([[c0_1, c0_2, c0_3, c0_4, c0_5]]).T 

        err_omega = sp.Matrix([[0,0,0,0,1]]) * mex * x0
        err_omega = err_omega[0, 0] # convert 1x1 matrix to scalar

        integration_bounds = (t, 0, sp.S.Infinity)

        # Quadratic part
        Pu_factor = self.matrix_2_norm(sp.Matrix(self.Minv.T @ self.R @ self.Minv))
        print(f"Calculation time Pu_factor: {time.time() - calculation_time} seconds")

        self.quadr_cost_x = sp.integrate(mex.T * self.Q  * mex, integration_bounds)
        print(f"Calculation time terminal cost (first quadratic part): {time.time() - calculation_time} seconds")
        self.quadr_cost_u = sp.integrate(mex.T * self.K.T @ self.K  * mex, integration_bounds)
        print(f"Calculation time terminal cost (second quadratic part): {time.time() - calculation_time} seconds")

        # Nonlinear part
        f_e = err_omega*r*(err_omega + 2*omega_theta)
        self.nonlin_cost = sp.integrate(f_e**2, integration_bounds)
        self.cost_u = 2 * Pu_factor * (x0.T @ self.quadr_cost_u @ x0 + sp.Matrix([self.nonlin_cost]))

        # Save for debugging
        self.cost_x_quadr = sp.lambdify((c0_1, c0_2, c0_3, c0_4, c0_5, r, omega_theta), self.quadr_cost_x)
        self.cost_u_quadr = sp.lambdify((c0_1, c0_2, c0_3, c0_4, c0_5, r, omega_theta), self.quadr_cost_u)
        self.cost_nonlin_u = sp.lambdify((c0_1, c0_2, c0_3, c0_4, c0_5, r, omega_theta), self.nonlin_cost)
        self.cost_Pu_fact = sp.lambdify((r, omega_theta), Pu_factor)

        # Lambdifify the cost function
        self.P = (x0.T * self.quadr_cost_x * x0)[0,0] + 2 * Pu_factor * (self.nonlin_cost + (x0.T * self.quadr_cost_u * x0)[0,0])
        self.P_function = sp.lambdify((c0_1, c0_2, c0_3, c0_4, c0_5, r, omega_theta), self.P)
        print(f"Overall calculation time terminal cost: {time.time() - calculation_time} seconds")

    def calculate_terminal_set(self, trajectory=None):
        input_bounds = InputBounds(self.model)
        A, b = input_bounds.get_conv_hull() # A, b for complete force
        b = b.reshape(-1,1) - A @ self.model.faulty_input_simple # A, b, for control force

        upper_bound = cp.Variable()
        lam = cp.Variable()
        constraints = []

        def get_s_procedure_constraint(F1, g1, h1, F2, g2, h2, lam):
            """
            x^T F1 x + 2 g1 x + h1 <= 0    ==>    x^T F2 x + 2 g2 x + h2 <= 0
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

        r = self.model.spiral_params.r
        omega_theta = self.model.spiral_params.omega_theta

        subst = sp.lambdify(self.r, sp.Matrix(self.Minv), "numpy")
        Minv = subst(r)

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
            mat = np.zeros_like(self.K)
            mat[0,4] = -np.sin(alpha) * 2*r*omega_theta
            mat[1,4] =  np.cos(alpha) * 2*r*omega_theta
            c2s.append(np.linalg.norm(-self.K + mat))

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
        constraints.append(get_s_procedure_constraint(self.P, None, -upper_bound,
                                                        C, None, -c3.item()**2, 
                                                        lam))

        # Solve the problem
        objective = cp.Maximize(upper_bound)
        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.SCS)

        terminal_set = EllipticalTerminalConstraint(prob.value, self.P)
        # print(f"Terminal set: {terminal_set}")
        return terminal_set

    def load_terminal_cost(self):
        cost_fcn = get_cost_function(self.model.spiral_params.r, 
                                 self.model.spiral_params.omega_theta)
        def correctly_sorted_cost_fcn(c):
            # c: [c1, c2, c3, c4, omega, alpha]
            return cost_fcn(c[0:5])
        
        return correctly_sorted_cost_fcn

    def matrix_2_norm(self, A):
        """
        2 norm for symbolic matrices.
        """
        # use '*' to unpack the list
        return sp.Max( *(A.singular_values()) )

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

    def create_python_code_P(self, code_file="./controllers/cost_function.py"):
        """
        Get the python code of the terminal cost.
        """
        var_tuple = "(c0_1, c0_2, c0_3, c0_4, c0_5)"
        return python_code_from_sympy(self.nonlin_cost, self.quadr_cost_x, var_tuple, code_file)

def create_cost_function(model, tuning_file="./controllers/tuning.yaml", 
                        failure_case="spiraling_5", param_set="P1",
                        code_destination="./controllers/cost_function.py"):
    """
    All-in-one function to create the cost function.
    """
    ti = TerminalIncredients(model, tuning_file, failure_case, param_set)
    ti.calculate_terminal_cost()
    ti.create_python_code_P(code_destination)

if __name__ == "__main__":
    model = get_faulty_ff_spiral()
    # create_cost_function(model)
    r = model.spiral_params.r
    omega_theta = model.spiral_params.omega_theta

    term = TerminalIncredients(model)
    term.calculate_terminal_set()
    # term.calculate_terminal_cost()
    # term.create_python_code_P("./controllers/cost_function.py")

    # for i in range(20):
    #     c = np.random.rand(5)

    #     loaded_fcn = term.load_terminal_cost()
    #     print(f"Difference cost fcn from file and directly: {loaded_fcn(c) - term.P_function(*c, r, omega_theta)}")


    # ellipse = term.calculate_terminal_set(model.spiral_params)
    # print(ellipse)


"""
Instead of labdifying the function, the intermediate output (that would later be used in
lambdify) can be used. This allows to get the python code of the function.

Attention: The printer does not convert everything to non-sympy code. This might not work 
for all cases! But: there might be work-arounds. In this case, the assumption 'r and om_theta 
are real' were necessary: Otherwise, the code would have contained a conjugate function. Also,
the ti.P[0,0] is needed to convert the matrix to a scalar; otherwise the code would have
contained a 'DenseMatrix' object.

https://stackoverflow.com/questions/27304590/generate-python-code-from-a-sympy-expression
https://docs.sympy.org/latest/modules/printing.html#module-sympy.printing.lambdarepr
# python_code = sp.printing.lambdarepr.lambdarepr(ti.P[0,0])
# # Add text to make the code a python function
# python_code = f"def cost_function(c0_1, c0_2, c0_3, c0_4, c0_5, c0_6, r, omega_theta):\n    return {python_code}"
# python_code = f"# Eigenvalues of closed loop: {ti.des_eigvals}\n\nfrom math import sqrt\n" + python_code
# # The code may become to long for one line. Therefore, split it into multiple expressions that
# # are then added

# var_tuple = (self.c0_1, self.c0_2, self.c0_3, self.c0_4, self.c0_5, self.c0_6, self.r,\
#               self.omega_theta)
# return python_code_from_sympy(self.P[0,0], var_tuple, code_file)

# var_tuple = "(c0_1, c0_2, c0_3, c0_4, c0_5, c0_6, r, omega_theta)"
# return python_code_from_sympy(self.P[0,0], var_tuple, code_file)
"""