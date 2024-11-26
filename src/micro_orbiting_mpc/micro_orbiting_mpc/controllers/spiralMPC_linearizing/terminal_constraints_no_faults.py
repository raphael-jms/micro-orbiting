import numpy as np
import scipy.linalg as la
import cvxpy as cp
import casadi as ca

from micro_orbiting_mpc.models.ff_input_bounds import InputBounds
from micro_orbiting_mpc.util.utils import read_yaml_matrix, read_yaml, EllipticalTerminalConstraint

def get_terminal_constraints(model):
    """
    Calculate the terminal incredients according to Chen and Allgöwer (1998).
    """
    # Define linearized system
    A = np.vstack((  np.hstack((np.zeros((3,3)), np.eye(3))),
                    np.zeros((3, 6))))
    mass = model.mass
    J = model.J
    B = np.vstack((np.zeros((3,3)),np.diag([1/mass, 1/mass, 1/J])))

    # **Step 1:** Solve linear control problem
    K = get_linearized_feedback(A, B)
    A_K = A + B @ K

    # **Step 2:** Determine kappa

    eig_val, eig_vec = la.eig(A_K)
    cond_kappa = -np.max(np.real(eig_val))

    kappa = 0.1
    if kappa > cond_kappa:
        raise ValueError(f"Choose kappa = {kappa} <= {cond_kappa}!")

    # **Step 3:** Find largest $\alpha$ s.t. $Kx \in U$ for all $x \in \Omega_{\alpha} = \{x \in R^n \vert x^T P_{cost} x \leq \alpha \}$

    alpha, P_cost = get_state_set_satisfying_input_constraints(A_K, K, kappa, model)

    # **Step 4:** Find the largest $\alpha_1 \in (0, \alpha]$ s.t. the following equation is satisfied: $$\sup \left\{ \frac{\Vert \Phi(x) \Vert}{\Vert x \Vert} \vert x \in \Omega_{\alpha}, x \neq 0 \right\} = L_{\Phi} \leq \frac{\kappa \lambda_{\min}(P)}{\Vert P \Vert}$$
    # Use the procedure described in Remark 3.1:
    # - make iterations of the optimization problem $$\underset{m}{\max} \{x^T P \Phi(x) - \kappa x^T P x \vert x^T P x \leq \alpha \}$$
    # - reduce the value of $\alpha$ until the optimal value is nonpositive
    opt_val = 1
    nb_iterations = 0
    while opt_val > 0:
        opt_val = solve_iterative_optimization_problem(P_cost, alpha, kappa, A_K, K, mass, J)
        if opt_val > 0:
            alpha = alpha * 0.9
            nb_iterations += 1
        if nb_iterations > 1:
            print(f"Number of iterations: {nb_iterations}, alpha: {alpha}, opt_val: {opt_val}")

    return EllipticalTerminalConstraint(alpha, P_cost)

def get_linearized_feedback(A, B):
    # penalize alpha a lot → staying within close reagion for alpha <=> bigger region for rest
    # Penalize control input a lot → Bigger resulting region with possible control input
    Qx_lin_fb = np.diag([1,1,100,1,1,1])
    Qu_lin_fb = np.diag([1,1,1]) * 10
    # [P_lin_fb, _, _] = control.care(A, B, Qx_lin_fb, Qu_lin_fb)
    P_lin_fb = la.solve_continuous_are(A, B, Qx_lin_fb, Qu_lin_fb)
    K = - la.inv(Qu_lin_fb) @ (B.T) @ P_lin_fb

    return K

def get_state_set_satisfying_input_constraints(A_K, K, kappa, model):
    tuning_file = "controllers/tuning.yaml"
    Qx = read_yaml_matrix(tuning_file, "faultfree", "P1", "Q")
    Qu = read_yaml_matrix(tuning_file, "faultfree", "P1", "R")

    # P_cost = la.solve_continuous_lyapunov( (A_K + kappa * np.eye(A_K.shape[0])).T , - (Qx + K.T @ Qu @ K) )
    P_cost = la.solve_continuous_are( A_K + kappa * np.eye(A_K.shape[0]), np.zeros((6,1)), Qx + K.T @ Qu @ K, 1 )
    # Equivalent statements

    bounds = InputBounds(model)

    # Since the condition has to be satisfied for all x in Omega, use the s procedure to convert 
    # the condition to a convex optimization problem

    # input constrains in form Mu <= m
    M, m = bounds.get_conv_hull()
    num_constraints = M.shape[0]

    alpha = cp.Variable((1,1))
    lambdas = [cp.Variable() for i in range(num_constraints)]

    constraints = []
    for i in range(num_constraints):
        lam = lambdas[i]
        m_i = m[i]
        MK_i = (M@K)[i,:].reshape((1,-1))

        n = P_cost.shape[0]
        mat1 = cp.vstack((
            cp.hstack((P_cost, np.zeros((n,1)))),
            cp.hstack((np.zeros((1,n)), -alpha))
        ))

        mat2 = cp.vstack((
            cp.hstack((np.zeros((n,n)), 0.5*MK_i.T)),
            cp.hstack((0.5*MK_i, np.array([[-m_i]])))
        ))

        constraints.append(mat1 - lam * mat2 >> 0)
        constraints.append(lam >= 0)

    constraints.append(alpha >= 0)

    objective = cp.Maximize(alpha)
    problem = cp.Problem(objective, constraints)
    # x = problem.solve(solver=cp.CLARABEL)
    x = problem.solve(solver=cp.SCS)

    alpha = alpha.value.item()

    return alpha, P_cost

# Check the maximum alpha_state contained in the found set
# Sanity check so that the set is not too big in alpha direction
def biggest_alpha_state(upper_bound, set_matrix):
    opti = ca.Opti()
    x = opti.variable(6,1)
    alpha_state = x[2]

    opti.minimize(-alpha_state)
    opti.subject_to(x.T @ set_matrix @ x <= upper_bound)

    opts = {
        "ipopt.print_level": 0, 
        "print_time": 0,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-8
    }
    opti.solver('ipopt', opts)
    sol = opti.solve()

    print(f"Biggest possible alpha (state): {sol.value(alpha_state)} rad = {sol.value(alpha_state)*180/np.pi}°, best x: {sol.value(x).T}")

    return sol.value(alpha_state)

def solve_iterative_optimization_problem(P_cost, alpha, kappa, A_K, K, mass, J):
    opti = ca.Opti()
    # Define optimization variables
    x = opti.variable(6,1)

    # Define nonlinear dynamics
    R = ca.MX(3,3)
    alpha_state = x[2]
    R[0,0] = ca.cos(alpha_state)/mass
    R[0,1] = -ca.sin(alpha_state)/mass
    R[1,0] = ca.sin(alpha_state)/mass
    R[1,1] = ca.cos(alpha_state)/mass
    R[2,2] = 1/J

    f = ca.MX(6,1)
    f[0:3] = x[3:6]
    f[3:6] = R @ K @ x

    Phi = f - A_K @ x

    # Define objective and constraints
    objective = x.T @ P_cost @ Phi - kappa * x.T @ P_cost @ x

    opti.minimize(-objective)
    opti.subject_to(x.T @ P_cost @ x <= alpha)

    opti.set_initial(x, [0,0.01,0,-0.01,0,0])

    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt', opts)

    # Solve the optimization problem
    sol = opti.solve()

    # Retrieve the optimal solution
    x_opt = sol.value(x)
    return sol.value(objective)
