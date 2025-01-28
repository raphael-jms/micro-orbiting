import numpy as np
from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull


dt = 0.1
params = dict(mass = 14.5, inertia = 0.37, max_force = 1.0, thruster_distance_to_center = 0.1)

model_fault_free = FreeFlyerDynamicsFull(dt, params)

model_first_stuck = FreeFlyerDynamicsFull(dt, params)
model_first_stuck.add_actuator_fault([1,1], 1)

model_second_stuck = FreeFlyerDynamicsFull(dt, params)
model_second_stuck.add_actuator_fault([1,2], 1)

model_both_stuck = FreeFlyerDynamicsFull(dt, params)
model_both_stuck.add_actuator_fault([1,1], 1)
model_both_stuck.add_actuator_fault([1,2], 1)

bounds_fault_free = InputBounds(model_fault_free)
ih_fault_free = InputHandlerImproved(model_fault_free, bounds_fault_free)

bounds_first_stuck = InputBounds(model_first_stuck)
ih_first_stuck = InputHandlerImproved(model_first_stuck, bounds_first_stuck)

bounds_second_stuck = InputBounds(model_second_stuck)
ih_second_stuck = InputHandlerImproved(model_second_stuck, bounds_second_stuck)

bounds_both_stuck = InputBounds(model_both_stuck)
ih_both_stuck = InputHandlerImproved(model_both_stuck, bounds_both_stuck)

mbis = [[model_fault_free, bounds_fault_free, ih_fault_free],
       [model_first_stuck, bounds_first_stuck, ih_first_stuck],
       [model_second_stuck, bounds_second_stuck, ih_second_stuck],
       [model_both_stuck, bounds_both_stuck, ih_both_stuck]]

u_des = np.array([0.5,0.0,0.0])

print("Desired input: ", u_des)
print("")

for mbi in mbis:
    model = mbi[0]
    bounds = mbi[1]
    ih = mbi[2]
    max_f = model.max_force

    u = ih.get_physical_input(u_des)

    print("Input full:   ", u/max_f)
    print("Input simple: ", model.u_full2u_simple_np @ u/max_f)
    print("Input faulty: ", model.faulty_input_simple.flatten()/max_f)
    print("")


