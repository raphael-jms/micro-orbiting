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

> [!TIP]
> For linearized spiraling MPC, to guarantee feasibility, the initial position can be set to
> ```
> export PX4_GZ_MODEL_POSE="0,-0.333,0.2,0,0,0"
> ```
> or the desired setpoint can be slightly altered to fit the center point better by setting `traj_shape: "hover_0_0.333_0"`.

## Setup for the MPC controller where the terminal controller is based on eMPC
In order to run the controller, the cost functions and terminal sets need to be pre-calculated first. This can be done calling
```
python3 -m micro_orbiting_mpc.util.setup_cost_fcn_db
```
from `~/micro-orbiting/src/micro_orbiting_mpc`. The resulting database is saved in `~/.ros/micro_orbiting/spiralMPC_empc_cost.db`.

## Available parameters in config dir:

> [!IMPORTANT]
> In order to make changed parameters available to ROS, the package nedds to be rebuilt using
> `colcon build`, respectively `colcon build --packages-select micro_orbiting_mpc`!

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
- For case spiralMPC_linearizing: 
    - recalculate_terminal_ingredients: True/False; otherwise last calculation is loaded from file

### Trajectory params
- traj_shape: generate_point_stabilizing, generate_sin, generate_line, generate_polynomial, generate_circle
- traj_duration: int
- Actuator faults: List of [[actuator], intensity_between_0_to_1, start_time]
                Only use if faults should be present at start

# ToDos

**Now**
- [x] Check terminal constraint in spiral linearizing MPC!!!
- [ ] Check proof for derivation of spiraling MPC!!! 
        - see comment in terminal_incredients_linearizing!!!
        - Why do I have `alpha/3` in spiral_mpc_1, build_solver???
- [ ] remove `recalculate_terminal_ingredients` from all files && check README


TODO add a timer for the calculation of the terminal sets

**Next steps**
- [x] Spiraling node
    - [x] Terminal constraint
- [x] add config file for the robot
- [x] Implement eMPC spiraling
    - [x] Necessary callers
    - [x] Terminal constraint
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
- [ ] The field `traj_duration` is completely ignored anyways
- [ ] Rename classes 
        - InputHandlerImproved to ControlAllocator
        - max_force to max_thrust
        - incredient to ingredient

**Other**
- [ ] Delete `sim_launch.py`? What exactly does it do?
- [ ] Add normalization of the error again?
- [ ] Generalization to 3D?
- [ ] Add a setup.py file where the db is calculated?

**Done**
- [x] how to load trajectories?
- [x] Create a damage simulation node
- [x] Find a solution to the jump of the angle for the MPC
- [x] Check again with the bounds for the optimization in basic MPC!!!

# Notes

## Install PlotJuggler:
```
sudo snap install plotjuggler
sudo apt install ros-$ROS_DISTRO-plotjuggler-ros
```
and run with
```
ros2 run plotjuggler plotjuggler
```

## Simulationn parameters Gazebo
```
~/.gz/fuel/fuel.gazebosim.org/proque/models/kth_freeflyer/1/model.sdf
```

# Install pympc
_Attention_: This is _not_ the `pympc` package from pip, but the one with the same name by [Tobia Marcucci](https://www.ece.ucsb.edu/people/faculty/tobia-marcucci). Install with
```
git clone git@github.com:TobiaMarcucci/pympc.git
pip install ./pympc
```
