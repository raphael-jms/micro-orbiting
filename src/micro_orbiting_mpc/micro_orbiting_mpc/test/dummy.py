import numpy as np

from micro_orbiting_mpc.controllers.controller_base_class import ControllerBaseClass
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved

class DummyModel:
    def __init__(self):
        self.name = 'DummyModel'

class DummyController(ControllerBaseClass):
    """
    Dummy controller for testing purposes
    Assumes that no failures have occured
    """
    def __init__(self):
        self.name = 'DummyController'
        self.model = FreeFlyerDynamicsFull(0.1)
        self.input_bounds = InputBounds(self.model)
        self.input_handler = InputHandlerImproved(self.model, self.input_bounds)

        self.k = 1

    def get_control(self, x0, t):
        control_x = -self.k * x0[0]
        control_y = -self.k * x0[1]
        control_alpha = -self.k * x0[2]

        u = np.array([control_x, control_y, control_alpha]).flatten()
        print(u)
        return self.input_handler.get_physical_input(u)