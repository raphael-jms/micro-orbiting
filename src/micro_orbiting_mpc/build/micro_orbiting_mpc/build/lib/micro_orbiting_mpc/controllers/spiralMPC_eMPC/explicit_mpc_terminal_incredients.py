import numpy as np
import sympy as sp
import casadi as ca
import matplotlib.pyplot as plt
import itertools
import subprocess
import scipy.linalg as la
import yaml

from micro_orbiting_mpc.util.utils import read_yaml_matrix, read_yaml
from micro_orbiting_mpc.util.polytope import MyPolytope
from micro_orbiting_mpc.models.ff_input_bounds import SpiralParameters, InputBounds
from micro_orbiting_mpc.util.utils import Rot3, Rot3Inv

from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from pympc.control.controllers import ModelPredictiveController

RobotToCenterRot = Rot3Inv 
CenterToRobotRot = Rot3

def sort_all_columns_right_to_left(arr):
    """
    Sort a 2D numpy array by all columns, starting from the rightmost column.
    
    Args:
        arr: 2D numpy array
        
    Returns:
        Sorted array
    """

    arr = np.round(arr, 4)

    for i in range(arr.shape[1]-1, -1, -1):
        arr = arr[arr[:, i].argsort(kind='stable')]
    return arr

"""
Issues and ToDos:
- calculate the largest cylinder in the polygon instead of underapproximating it as the largest
cylinder in the largest square in the polygon
- would be better to implement empc as a 4D-system.
- make c artificially small in bounding? Remove complete?

"""
class explicitMPCTerminalIngredients:
    def __init__(self, model, tuning):
        """
        Calculate the terminal ingredients based on an explicit MPC controller 
        as part of the terminal controller.
        """
        self.model=model
        self.spiral_params = SpiralParameters(model)

        self.max_acceleration = tuning["max_acceleration"]
        self.k_omega = tuning["k_omega"]
        self.tuning = tuning

        self.r = self.spiral_params.r
        self.omega_theta = self.spiral_params.omega_theta
        self.virt_force = ca.DM(self.spiral_params.b)

        self.mass = model.mass
        self.J = model.J

        # The cost matrices for the function used in the MPC; Used to calculate terminal cost
        self.Q = np.diag(tuning["Q"])
        self.R = np.diag(tuning["R"])
        self.ensure_diagonal(self.Q)
        self.ensure_diagonal(self.R)

        self.Minv = np.array([[self.mass, 0, self.r*self.mass],
                              [0, self.mass, 0],
                              [0, 0, self.J]])

        self.M = np.array([[1/self.mass, 0, -self.r/self.J], 
                           [0, 1/self.mass, 0],
                           [0,0,1/self.J]])

    def calculate_empc_input_bounds(self):
        bounds = InputBounds(self.model)
        J = self.J; r = self.r; omega_th = self.omega_theta; k_omega = self.k_omega
        b_force = RobotToCenterRot(self.spiral_params.beta - np.pi/2) @ self.virt_force # Calculate in virt_force-aligned local system

        # First step: Take acceleration into account
        A, b = bounds.get_conv_hull()
        A = A @ CenterToRobotRot(self.spiral_params.beta - np.pi/2) # Calculate in virt_force-aligned local system

        # transform to accelerations
        P_full = MyPolytope(A @ self.Minv, b)
        P_offsetfree = P_full.minkowski_subtract_circle(self.max_acceleration)

        # Second step: Calculate u3max
        opti = ca.Opti()
        u = opti.variable(P_offsetfree.Nx)

        constraints = []
        constraints.append(u >= np.zeros(P_offsetfree.Nx))

        # create corner points of a box centered at zero with edge_length = 2u
        combinations = list(itertools.product(*[[-u[i], u[i]] for i in range(P_offsetfree.Nx)])) 
        u3 = u[2]
        b_of_u = P_offsetfree.b - np.sign(P_offsetfree.b) * (1/k_omega * u3 + 2*omega_th) * 1/k_omega * r * u3 

        for combination in combinations:
            constraints.append(
                ca.mtimes(P_offsetfree.A, self.M @ b_force + ca.vertcat(*combination)) <= b_of_u
            )

        opti.subject_to(constraints)

        obj = 0
        for i in range(P_offsetfree.Nx):
            obj += ca.log(u[i])

        opti.minimize(-obj)

        opti.solver('ipopt', {}, {"print_level": 0})
        opti.set_initial(u,  np.ones(A.shape[1]))
        sol = opti.solve()

        return sol.value(u)

    def calculate_terminal_set(self, xSet, ySet):
        """ 
        Calculate the terminal set for the terminal controller.  The result is the union of all 
        regions of the eMPC controller.
        """
        assert xSet.Nx == 2 and ySet.Nx == 2

        # Separate 2-dim (1-dim) sets for x-, y- and alpha-direction: combine into one 5-dim set
        A = np.vstack((
            np.hstack((  xSet.A[:,0].reshape(-1,1), np.zeros((xSet.Nc, 1)), xSet.A[:,1].reshape(-1,1), np.zeros((xSet.Nc, 2))  )),
            np.hstack((  np.zeros((ySet.Nc, 1)), ySet.A[:,0].reshape(-1,1), np.zeros((ySet.Nc, 1)), ySet.A[:,1].reshape(-1,1), np.zeros((ySet.Nc, 1))  )),
            np.array([[0,0,0,0, 1],
                      [0,0,0,0,-1]]) 
        ))
        b = np.vstack((
            xSet.b.reshape(-1,1),
            ySet.b.reshape(-1,1),
            1/self.k_omega * self.u3max, 
            1/self.k_omega * self.u3max 
        ))

        return MyPolytope(A, b)

    def ensure_diagonal(self, matrix):
        """
        Ensure that the matrix is diagonal.
        """
        if np.any( np.abs(matrix - np.diag(np.diag(matrix))) >= 0.0001* np.ones_like(matrix)):
            raise ValueError("Matrix is not diagonal.")

    def matrix_2_norm(self, A):
        """
        2 norm for symbolic matrices.
        """
        # use '*' to unpack the list
        return sp.Max( *(A.singular_values()) )

    def calculate_empc(self, uimax, horizon, time_scaling = 5):
        """
        Calculate the explicit MPC controller. Calculating the explicit solution for a LTI 
        system is computationally quite expensive. In order to get large terminal sets, one can
        therefore either increase the computation time (with a very bad scaling) or resample 
        the system: n-times larger time step → terminal set becomes larger (faster large). This
        is equivalent to having the original dt and always applying the same control n times in
        a row. Therefore, feasibility and stability are still ensured. In order to get the 
        same value-function, the cost matrices need to be scaled: Q_resampled = n*Q_original.
        This is only an optimization step to ensure that the calculations do not take too long.
        """
        # Continuous-time dynamics
        A = np.array([[0,1],[0,0]])
        B = np.array([[0, 1]]).T

        # Discrete-time dynamics
        sys = LinearSystem.from_continuous(A, B, time_scaling*self.model.dt, 'zero_order_hold')

        if self.R[0,0] != self.R[1,1]:
            raise ValueError("With the chosen Qu_bar, the first two elements of R need to be equal.")
        R = np.array(self.R[0,0])**2 * self.r**2 * self.mass**4 + np.array(self.R[0,0]) * self.mass**2
        if self.Q[0,0] != self.Q[1,1] or self.Q[2,2] != self.Q[3,3]:
            raise ValueError("Not mathematically necessary, but the given implementation " + 
                             "requires Q[0,0] = Q[1,1] and Q[2,2] = Q[3,3].")
            # otherwise, the empc controller needs to be calculated twice with the different parameters
        Q = np.array([[self.Q[0,0], 0],[0, self.Q[2,2]]])
        # Adapt Q and R to the time scaling
        R *= time_scaling
        Q *= time_scaling
        
        # Terminal controller for the explicit MPC
        P, K = sys.solve_dare(Q, R)

        # State and input constraints
        U = Polyhedron.from_bounds(-uimax, uimax)
        X = Polyhedron.from_bounds(np.array([-5, -1.5]), # TODO parametrize
                                   np.array([ 5,  1.5]))
        D = X.cartesian_product(U)

        # Terminal set for the explicit MPC
        X_N = sys.mcais(K, D)

        # Calculate controller
        controller = ModelPredictiveController(sys, horizon, Q, R, P, D, X_N)
        controller.store_explicit_solution(verbose=True)

        return controller

    def bound_empc_cost(self, empc):
        # Get the vertices of all controlled sets
        allvertices = []
        for active_set in empc.explicit_solution.critical_regions:
            allvertices.extend(active_set.polyhedron.vertices)

        covered_area = MyPolytope.from_vertices(allvertices)
        allvertices = np.array(allvertices).T

        # reduce the calculation effort # TODO parametrize this somehow
        bounds_min = np.array([-5,-5])
        bounds_max = np.array([ 5, 5])

        # Get tuples of the bounds and step size in each dimension
        slices = [slice(start, stop, 0.1) for start, stop in zip(bounds_min, bounds_max)]

        # Get all points in the space
        points = np.mgrid[slices].reshape(2,-1).T

        opti = ca.Opti()

        a = opti.variable(empc.S.nx, empc.S.nx, 'symmetric') # A needs to be positive (semi)-definite
        b = opti.variable(empc.S.nx)
        c = opti.variable(1)

        conditions = []
        obj = 0

        for point in points:
            val = empc.explicit_solution.V(point)
            if val is None:
                continue
            val = ca.DM(val)
            point = ca.DM(point)
            opt = point.T @ a @ point + b.T @ point + c

            conditions.append(val <= opt)
            obj += (opt-val)**2
            # obj += opt**2

        # conditions.append( c <= 0.05 )

        opti.subject_to(conditions)
        opti.minimize(obj)
        opti.solver('ipopt')

        # opti.set_initial(a, controller.explicit_solution.critical_regions[0]._V['xx'])
        opti.set_initial(b, empc.explicit_solution.critical_regions[0]._V['x'])
        opti.set_initial(c, 0)

        sol = opti.solve()
        terminal_cost = {'xx': sol.value(a), 'x': sol.value(b), 'c': sol.value(c)}
        print(f"xx: {terminal_cost['xx']}, x: {terminal_cost['x']}, c: {terminal_cost['c']}")
        return terminal_cost, covered_area

    def calculate_terminal_ingredients(self, calculate_empc=True):
        # Calculate the bounds for the linear controllers (omega & empc)
        u_bound = self.calculate_empc_input_bounds()
        self.u3max =  u_bound[2]
        self.u3min = -u_bound[2]
        # u1max = u_bound[0]/np.sqrt(2)
        # u2max = u_bound[1]/np.sqrt(2)
        # uimax = max(u1max, u2max)

        # Calculate the maximal box bound (edge distance from 0) based on the vertices
        ang = np.arctan2(u_bound[1], u_bound[0])
        rad = np.sqrt(u_bound[0]**2 + u_bound[1]**2)
        u1max = np.cos(ang)*rad
        u2max = np.sin(ang)*rad
        uimax = max(u1max, u2max)

        print(f"u_bound: {u_bound}")

        # Calculate the explicit MPC and cost
        empc_horizon = self.tuning["empc_horizon"]
        time_scaling = self.tuning["time_scaling"]
        empc = self.calculate_empc(uimax, empc_horizon, time_scaling)

        # calculate bound and terminal set for respective layer
        costs, sets = [], []
        t_cost, t_set = self.bound_empc_cost(empc)
        # appended twice as the same cost is used for both x- and y-direction
        costs.append(t_cost)
        costs.append(t_cost)
        sets.append(t_set)
        sets.append(t_set)

        # Calculate the complete cost
        e0_1, e0_2, e0_3, e0_4, e0_5 = sp.symbols('e0_1 e0_2 e0_3 e0_4 e0_5')

        # cost_empc
        err_x = sp.Matrix([e0_1, e0_3])
        err_y = sp.Matrix([e0_2, e0_4])
        assert np.abs(costs[0]['c']) < 0.0001 # Actually, it should be zero
        cost_empc  = err_x.T @ sp.Matrix(costs[0]['xx']) @ err_x + sp.Matrix(costs[0]['x']).T @ err_x #+ sp.Matrix([[costs[0]['c']])
        cost_empc += err_y.T @ sp.Matrix(costs[1]['xx']) @ err_y + sp.Matrix(costs[1]['x']).T @ err_y #+ sp.Matrix([[costs[1]['c']])
        cost_empc = cost_empc[0, 0] # make scalar


        print("cost xx, x, c")
        print(costs[0]['xx'])
        print(costs[0]['x'])
        print(costs[0]['c'])

        q1, q2, qomega = self.Q[0, 0], self.Q[1, 1], self.Q[4, 4]

        # cost_e5
        A_e5_subsystem = np.array([[1 - self.k_omega * self.model.dt]])
        P_e5_subsystem = la.solve_discrete_lyapunov(A_e5_subsystem, np.array([[qomega]])).item()
        cost_e5 = e0_5**2 * P_e5_subsystem

        print(f"cost e5: {cost_e5}")
        # cross term
        approx_abs_e0_5 = e0_5 * sp.tanh(e0_5/0.1)
        cross_term_constant = 2 * np.sqrt(u1max**2 + u2max**2) * self.mass**2 * q1 * self.r
        cross_term_var = e0_5**2 / ( 1 - (1-self.k_omega)**2 ) \
                        + (self.k_omega + 2 * sp.Abs(self.omega_theta)) / self.k_omega * approx_abs_e0_5
        cost_cross_term = cross_term_constant * cross_term_var
        
        print(f"cost cross: {cost_cross_term}")
        # feedback term
        fb_factor = self.matrix_2_norm(sp.Matrix(self.Minv.T @ self.R @ self.Minv))
        P_K = la.solve_discrete_lyapunov(A_e5_subsystem, np.array([[1]])).item()
        cost_fb_lin = 2 * fb_factor * (
            e0_5**2 * P_K
            + (2*self.omega_theta*self.r)**2 / (1 - (1-self.k_omega)**2 ) * e0_5**2
            + (4*self.omega_theta*self.r) / ( 1 - (1-self.k_omega)**3 ) * e0_5**3
            + self.r**2 / ( 1 - (1-self.k_omega)**4 ) * e0_5**4
        )
        print(f"cost cost_cross_term: {cost_fb_lin}")


        self.terminal_cost = cost_empc + cost_e5 + cost_cross_term + cost_fb_lin
        self.terminal_cost_function = sp.lambdify((e0_1, e0_2, e0_3, e0_4, e0_5), self.terminal_cost)

        print("terminal cost overall")
        print(self.terminal_cost_function)
        ## Calculate the terminal sets
        self.terminal_set = self.calculate_terminal_set(*sets)

    def create_python_code(self, cost):
        """
        Make the sympy code normal Python code and save it
        """
        python_code = sp.python(cost)

        for line in python_code.split("\n"):
            if line.startswith("e = "):
                final_expression = line.replace("e = ", "")
                break

        code = f"sp.lambdify((e0_1, e0_2, e0_3, e0_4, e0_5), {final_expression}, modules={{'Abs':ca.fabs, 'tanh':ca.tanh}})"
        # execute code with cost = eval()
        return code

    def get_terminal_cost(self):
        if self.terminal_cost is None:
            raise ValueError("Terminal cost not calculated.")
        return self.create_python_code(self.terminal_cost)

    def get_terminal_set(self):
        if self.terminal_set is None:
            raise ValueError("Terminal set not calculated.")
        return self.terminal_set

    def compare_bound_with_controller(self, empc, bound):
        """
        Compare the combined upper bound for all terminal regions  with the actual terminal cost
        for the eMPC controller
        """
        if empc.Nx != 2:
            Warning("Only 2D plots are supported.")
            return

        xymax = 5
        # xymax = 15
        X = np.arange(-xymax, xymax, 0.15)
        Y = np.arange(-xymax, xymax, 0.15)
        X, Y = np.meshgrid(X, Y)

        def f(x, y):
            state = np.array([x, y])
            return empc.V(state)

        f_vec = np.vectorize(f)

        # Calculate Z values
        Z = f_vec(X, Y)

        # Create the 3D plot
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = axs[0].plot_surface(X, Y, Z, cmap='viridis')

        # Add a color bar
        fig.colorbar(surf)

        # Set labels
        axs[0].set_xlabel('x1')
        axs[0].set_ylabel('x2')
        axs[0].set_zlabel('V')

        def bound_function(x, y):
            state = np.array([x, y]).reshape((-1,1))
            return state.T @ bound['xx'] @ state + bound['x'].T @ state + bound['c']

        bound_vec = np.vectorize(bound_function)

        # Calculate Z values
        Z_bound = bound_vec(X, Y)
        Z_bound[Z == None] = None
        Z_bound[Z_bound == np.nan] = None
        surf_bound = axs[0].plot_surface(X, Y, Z_bound, cmap='inferno')
        Z_bound[np.isnan(Z_bound)]= 0

        # Second plot: errors
        # Create the 3D plot
        Zcopy = Z.copy()

        Z_bound[Z == None] = 0
        Z[Z == None] = 0

        Diff = Z - Z_bound
        Diff[Zcopy == None] = None
        # Plot the surface
        surf = axs[1].plot_surface(X, Y, Diff, cmap='cividis_r')
        
        # Plot zero
        surf = axs[1].plot_surface(X, Y, np.zeros_like(Diff), alpha=0.1)


        # Set labels
        axs[0].set_xlabel('x1')
        axs[0].set_ylabel('x2')
        axs[1].set_zlabel('Error')

        # axs[1].set_zlim3d(bottom=np.min(Diff), top = np.max(Diff))

        axs[1].set_title('Error between V and bound')

        plt.show()

    def plot_empc_bounds(self, u_empc):
        bounds = InputBounds(self.model)
        A, b = bounds.get_conv_hull()
        P_full = MyPolytope(A, b)
        P_offsetfree = P_full.minkowski_subtract_circle(self.max_acceleration)
        P_empc = MyPolytope.from_box(-u_empc, u_empc)
        P_empc = MyPolytope.from_box(-u_empc+self.spiral_params.b, u_empc+self.spiral_params.b)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        P_full.plot(ax, 'b')
        P_offsetfree.plot(ax, 'r')
        P_empc.plot(ax, 'g')

        plt.show()

if __name__ == "__main__":
    from micro_orbiting_mpc.util.get_example import get_faulty_ff_full

    sys = get_faulty_ff_full()
    tuning = {
        "max_acceleration" : 0,
        "k_omega" : 1,
        "Q" : [1, 1, 1, 1, 1],
        "R" : [1, 1, 1]
    }
    ti = explicitMPCTerminalIngredients(sys, tuning)
    ti.calculate_terminal_ingredients(calculate_empc=False)
    code = ti.create_python_code(ti.terminal_cost)

    breakpoint()
