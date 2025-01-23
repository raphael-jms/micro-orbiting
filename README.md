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

```
vim /home/raphael/PX4-Space-Systems/build/px4_sitl_default/rootfs/etc/init.d-posix/airframes/71002_gz_spacecraft_2d
```

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
- [ ] implement reactive mode
    - [ ] Write reactive class
    - [ ] implement into spacecraft_mpc_node
    - [ ] Write code o calculate terminal cost for all cases
- [ ] Remove the parameter handling again?



- [ ] add code to handle actuator failures in 0 /in /Interior case
    - [ ] add compensation function
    - [ ] different actuator limits
    - [ ] control allocation? What about system fcn?


- start filtering actuator failures for start time
    -> Only apply 0 ones directly
    -> do this in both fault_simulation_node and spacecraft_mpc_node
- behavior for fault_simulation_node :
    1. apply the 0 ones directly
    2. Listen to manually added ones
    3. apply the non-0 ones from the launch file only if no manually ones have been added
        *Alternative*: Modify the model so that it can handle changes in failures










**Next steps**
- [ ] Not sure when: Add new arena to Gazebo simulation?
- [ ] Implement reactive mode
        → should probably be part of the controller classes?
            → from the idea, it does not belong to ROS
            → Not sure about the models etc.

**Code clean up**
- [ ] Remove option for `trajectory_tracking=False`
- [ ] Decide what to do with alternative to `spiral_5` (linearizing Spiral MPC)
- [ ] remove `recalculate_terminal_ingredients` from all files && check README
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
- [ ] Decide how to distribute: With or without database?

**Check for paper**
- [ ] Check: The resulting linearized system is pretty easy (double integrator)
        -> There should be some results concerning this system under input constraints
        -> Has anyone tried to use these results for calculating terminal sets?
        -> That might be computationally much more efficient
        -> Requirement is of course that a valid cost function is available
- [ ] For the spiralingMPC_eMPC cost, the bounding of the eMPC cost is done by sampling and overapproximating the eMPC cost. Is there a better way to do it?

**Done**
- [x] how to load trajectories?
- [x] Create a damage simulation node
- [x] Find a solution to the jump of the angle for the MPC
- [x] Fix rest of code s.t. eMPC runs
- [x] Add state publisher code for eMPC
- [x] Take care of linMPC terminal set [x] Check again with the bounds for the optimization in basic MPC!!!
- [x] Check terminal constraint in spiral linearizing MPC!!!
- [x] Spiraling node
    - [x] Terminal constraint
- [x] add config file for the robot
- [x] Implement eMPC spiraling
    - [x] Necessary callers
    - [x] Terminal constraint

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

## Simulation parameters Gazebo
```
~/.gz/fuel/fuel.gazebosim.org/proque/models/kth_freeflyer/1/model.sdf
```

## Install pympc
_Attention_: This is _not_ the `pympc` package from pip, but the one with the same name by [Tobia Marcucci](https://www.ece.ucsb.edu/people/faculty/tobia-marcucci). Install with
```
git clone git@github.com:TobiaMarcucci/pympc.git
pip install ./pympc
```

## Add gazebo worlds
Add gazebo worlds under
```
/home/raphael/PX4-Space-Systems/Tools/simulation/gz/worlds/
```
collada (mesh) files can be added in the mesh folder and referenced as in the myworld.sdf
Change world file using
```
vim /home/raphael/PX4-Space-Systems/build/px4_sitl_default/rootfs/etc/init.d-posix/airframes/71002_gz_spacecraft_2d
```

## If simulation is not starting after running before
```
ps aux | grep gz
> raphael     7122 10.1  1.0 1862636 171796 pts/2  Sl   16:07   1:16 gz sim --verbose=1 -r -s /home/raphael/PX4-Space-Systems/Tools/simulation/gz/worlds/default.sdf
> raphael    11689  0.0  0.0  21068  2432 pts/2    S+   16:19   0:00 grep --color=auto gz
kill -9 7122
```
