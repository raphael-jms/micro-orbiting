from models.ff_dynamics import FreeFlyerDynamicsFull, SpiralDynamics
from models.ff_input_bounds import InputBounds, InputHandlerImproved, SpiralParameters

def get_faulty_ff_full(dt=0.1):
    model = FreeFlyerDynamicsFull(dt)

    model.add_actuator_fault([3, 1], 1)
    model.add_actuator_fault([4, 1], 1)

    return model

def get_faulty_ff_spiral(dt=0.1):
    free_flyer = FreeFlyerDynamicsFull(dt)
    free_flyer.add_actuator_fault([3, 1], 1)
    free_flyer.add_actuator_fault([4, 1], 1)
    # spiral_params = SpiralParameters(free_flyer)

    # spiral_model = SpiralDynamics(dt, spiral_params)

    # return spiral_model
    return SpiralDynamics.from_ff_model(free_flyer)