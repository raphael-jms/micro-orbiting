import sympy as sp
from sympy import Symbol, Float, Max, sqrt
import numpy as np

def get_cost_function(r, omega_theta):
    c0_5 = Symbol('c0_5')
    
    c0_1 = Symbol('c0_1')
    c0_2 = Symbol('c0_2')
    c0_3 = Symbol('c0_3')
    c0_4 = Symbol('c0_4')
    c0_5 = Symbol('c0_5')
    P_1 = Float('4.2163702135578385',precision=53)* c0_5**3* omega_theta* r**2
    P_2 = Float('6.3245553203367573',precision=53)* c0_5**2* omega_theta**2* r**2
    P = Float('0.79056941504209466',precision=53)*c0_5**4*r**2+ P_1+ P_2
    cost_matrix_0_0 = 9.72404834564383
    cost_matrix_0_1 = 0
    cost_matrix_0_2 = 7.90569415042092
    cost_matrix_0_3 = 0
    cost_matrix_0_4 = 0
    cost_matrix_1_0 = 0
    cost_matrix_1_1 = 9.72404834564385
    cost_matrix_1_2 = 0
    cost_matrix_1_3 = 7.90569415042095
    cost_matrix_1_4 = 0
    cost_matrix_2_0 = 7.90569415042092
    cost_matrix_2_1 = 0
    cost_matrix_2_2 = 9.35423686048696
    cost_matrix_2_3 = 0
    cost_matrix_2_4 = 0
    cost_matrix_3_0 = 0
    cost_matrix_3_1 = 7.90569415042095
    cost_matrix_3_2 = 0
    cost_matrix_3_3 = 9.35423686048701
    cost_matrix_3_4 = 0
    cost_matrix_4_0 = 0
    cost_matrix_4_1 = 0
    cost_matrix_4_2 = 0
    cost_matrix_4_3 = 0
    cost_matrix_4_4 = 158.113883008419
    cost_matrix = np.zeros((5, 5))
    cost_matrix[0, 0] = cost_matrix_0_0
    cost_matrix[0, 1] = cost_matrix_0_1
    cost_matrix[0, 2] = cost_matrix_0_2
    cost_matrix[0, 3] = cost_matrix_0_3
    cost_matrix[0, 4] = cost_matrix_0_4
    cost_matrix[1, 0] = cost_matrix_1_0
    cost_matrix[1, 1] = cost_matrix_1_1
    cost_matrix[1, 2] = cost_matrix_1_2
    cost_matrix[1, 3] = cost_matrix_1_3
    cost_matrix[1, 4] = cost_matrix_1_4
    cost_matrix[2, 0] = cost_matrix_2_0
    cost_matrix[2, 1] = cost_matrix_2_1
    cost_matrix[2, 2] = cost_matrix_2_2
    cost_matrix[2, 3] = cost_matrix_2_3
    cost_matrix[2, 4] = cost_matrix_2_4
    cost_matrix[3, 0] = cost_matrix_3_0
    cost_matrix[3, 1] = cost_matrix_3_1
    cost_matrix[3, 2] = cost_matrix_3_2
    cost_matrix[3, 3] = cost_matrix_3_3
    cost_matrix[3, 4] = cost_matrix_3_4
    cost_matrix[4, 0] = cost_matrix_4_0
    cost_matrix[4, 1] = cost_matrix_4_1
    cost_matrix[4, 2] = cost_matrix_4_2
    cost_matrix[4, 3] = cost_matrix_4_3
    cost_matrix[4, 4] = cost_matrix_4_4
    
    
    nonlinear_cost = sp.lambdify((c0_1, c0_2, c0_3, c0_4, c0_5), P)
    quadratic_cost = cost_matrix

    def cost_function(x):
        return nonlinear_cost(x[0], x[1], x[2], x[3], x[4]) + x.T @ quadratic_cost @ x
    return cost_function
