# micro-orbiting

## How to use

Several launch files exist that start the controller with different functionality:
- Nominal case (no faults)
- ...

The launch files can be found in the `launch` folder and be started as
```
ros2 launch micro_orbiting_mpc spacecraft_mpc_node.nominal_mpc.py 
```

The parameters can be changed in the corresponding parameter file (for the case of the example above, `config/nominal_mpc.yaml`) and include
- MPC parameters (horizon, time step, ...)
- the trajectory
- MPC tuning (multiple parameter sets are possible and can be selected using the variable `param_set`, also in the config file)

The trajectories can be set by publishing to the topic `/trajectory_commands`. The one from the parameter file is published automatically at startup and will be overwritten with every new ROS trajectory message.

## Available parameters in config dir:

### MPC params
- mode: faultfree, reactive, spiralMPC_linearizing, spiralMPC_eMPC, dummy
- horizon: MPC horizon
- time_step: MPC sampling time
- trajectory_tracking: True/False
- *evtl* param mode name?
- param_set: P1, P2, ...
- solver_opts: Dict of solver options
- dt: MPC time step
- uub: input upper bounds (list of [x, y])
- ulb: input lower bounds (list of [x, y])
- xub: state upper bounds (list of [x, y])
- xlb: state lower bounds (list of [x, y])

### Trajectory params
- traj_shape: generate_point_stabilizing, generate_sin, generate_line, generate_polynomial, generate_circle
- traj_duration: int
- Actuator faults: List of [[actuator], intensity_between_0_to_1, start_time]
                Only use if faults should be present at start

# ToDos

- [ ] Check code for unnecessary lines / commented out stuff to be deleted / ...
- [ ] Decide how to distribute: With or without database?
- [ ] Generalization to 3D?
- [ ] Later
    - [ ] Rename classes (e.g. InputHandlerImproved to ControlAllocator, max_force to max_thrust)

- [ ] Delete `sim_launch.py`? What exactly does it do?

- [ ] Create a damage simulation node
- [ ] Find a solution to the jump of the angle for the MPC


- [x] how to load trajectories?