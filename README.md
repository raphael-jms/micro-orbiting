# micro-orbiting

## How to use

Several launch files exist that start the controller with different functionality:
- Nominal case (no faults)
- ...

The launch file can be found in the `launch` folder and be started as
```
ros2 launch micro_orbiting_mpc spacecraft_mpc_node.launch.py config_file:=nominal_mpc.yaml
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

**Now**
- [ ] Check again with the bounds for the optimization in basic MPC!!!
- [ ] Check terminal constraint in spiral linearizing MPC

**Next steps**
- [ ] Spiraling node
    - [ ] Terminal constraint
- [ ] Implement eMPC spiraling
    - [ ] Necessary callers
    - [ ] Terminal constraint
    - [ ] Decide how to distribute: With or without database?
- [ ] Not sure when: Add new arena to Gazebo simulation?
- [ ] Implement reactive mode
        → should probably be part of the controller classes?
            → from the idea, it does not belong to ROS
            → Not sure about the models etc.

**Code clean up**
- [ ] Remove option for `trajectory_tracking=False`
- [ ] Decide what to do with alternative to `spiral_5` (linearizing Spiral MPC)
- [ ] Check code for unnecessary lines / commented out stuff to be deleted / ...
- [ ] Rename classes 
        - InputHandlerImproved to ControlAllocator
        - max_force to max_thrust
        - incredient to ingredient

**Other**
- [ ] Delete `sim_launch.py`? What exactly does it do?
- [ ] Add normalization of the error again?
- [ ] Generalization to 3D?

**Done**
- [x] how to load trajectories?
- [x] Create a damage simulation node
- [x] Find a solution to the jump of the angle for the MPC

# Notes

Install PlotJuggler:
```
sudo snap install plotjuggler
sudo apt install ros-$ROS_DISTRO-plotjuggler-ros
```
and run with
```
ros2 run plotjuggler plotjuggler
```