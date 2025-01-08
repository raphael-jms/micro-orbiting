import numpy as np
import math

from micro_orbiting_mpc.controllers.controller_base_class import ControllerBaseClass
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved

class DummyModel:
    def __init__(self):
        self.name = 'DummyModel'
        self.max_force = 1.75

class DummyController(ControllerBaseClass):
    """
    Dummy controller for testing purposes
    Assumes that no failures have occured
    """
    def __init__(self, ros_node):
        super().__init__(ros_node)
        self.name = 'DummyController'
        self.model = FreeFlyerDynamicsFull(0.1)
        self.input_bounds = InputBounds(self.model)
        self.input_handler = InputHandlerImproved(self.model, self.input_bounds)

        self.k = 0.01
        self.d = 0.05

        self.dt = 0.1

    def get_control(self, x0, t):
        e_x = x0[0] - 3
        e_y = x0[1]

        control_x = -self.k * e_x - 0.1 * x0[3]
        control_y = -self.k * e_y - 0.1 * x0[4]
        # control_alpha = -self.k * x0[4]
        alpha = x0[2]

        [control_x, control_y] = list(
            # np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]).reshape(2,2) @ np.array([control_x, control_y])
            np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]).reshape(2,2) @ np.array([control_x, control_y])
        )

        # PD controller for alpha
        alpha = x0[2]
        omega = x0[5]
        control_alpha = -self.k * alpha - self.d * omega

        u = np.array([control_x, control_y, control_alpha]).flatten()
        return self.input_handler.get_physical_input(u)

    def assign_trajectory(self, traj):
        self.trajectory = "This controller does not support a trajectory"