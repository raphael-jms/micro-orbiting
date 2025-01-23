import threading
import pprint

from micro_orbiting_mpc.controllers.controller_mpc_base import GenericMPC
from micro_orbiting_mpc.controllers.spiralMPC_eMPC.controller_empc import FancyMPC
from micro_orbiting_mpc.controllers.fb_linearizing_controller import FBLinearizingController
from micro_orbiting_mpc.controllers.nominalMPC_no_faults.terminal_constraints_no_faults import get_terminal_constraints_no_faults

from micro_orbiting_mpc.models.ff_dynamics import SpiralDynamics
from micro_orbiting_mpc.models.ff_input_bounds import InputHandlerImproved, InputBounds
from micro_orbiting_mpc.controllers.state_constants import States
from micro_orbiting_mpc.util.utils import ensure_proper_fault_information

from micro_orbiting_msgs.msg import FailedActuator

class ReactiveController:
    """
    Reactive controller: Chooses the appropriate controller based on the current failure state
    """
    def __init__(self, model, params, robot_params, actuator_failures_plan, _ros_node):
        self._ros_node = _ros_node

        self.params = params
        self.robot_params = robot_params

        self.model = model
        self.spiral_model = None
        self.controller = None
        self.trajectory = None
        self.traj_start_time = None # keep track of state of trajectory across multiple controllers
        self.actuator_failures = [] # of type FailedActuator

        # handle actuator failures that are present at startup
        actuator_failures_at_start = []
        if actuator_failures_plan is not None:
            for act in actuator_failures_plan:
                if act["start_time"] == 0:
                    actuator_failures_at_start.append(FailedActuator(pos1=act["act_ids"][0], 
                                                                    pos2=act["act_ids"][1], 
                                                                    intensity=act["intensity"]))
            self.add_actuator_failures(actuator_failures_at_start)

        self.input_bounds = InputBounds(model)
        self.input_handler = InputHandlerImproved(self.model, self.input_bounds)

        self.initialization_lock = threading.Lock()
        self._set_controller()

    def _set_controller(self):
        match self.input_handler.get_fault_category():
            case States.ORIGIN_IN_INTERIOR_OF_U:
                # Using nominal controller

                if self.controller is not None: # No fallback needed at startup
                    # self._init_feedback_linearizing_controller()
                    self._ros_node.get_logger().warn("Keeps on using old controller until new one is loaded")
                
                    # Async initialization of spiraling MPC
                    init_thread = threading.Thread(target=self._init_nominal_MPC)
                    init_thread.start()
                else:
                    self._init_nominal_MPC()

            case States.ORIGIN_ON_BOUNDARY_OF_U | States.ORIGIN_OUTSIDE_OF_U:
                # Using fblin controller as quick fallback, then spiraling controller

                if self.controller is not None: # No fallback needed at startup
                    self._init_feedback_linearizing_controller()
                
                    # Async initialization of spiraling MPC
                    init_thread = threading.Thread(target=self._init_spiraling_MPC)
                    init_thread.start()
                else:
                    self._init_spiraling_MPC()

            case _:
                raise ValueError("Invalid fault category")

    def _init_feedback_linearizing_controller(self):
        tuning_param = self.params["tuning"]["fb_lin"]
        tuning = tuning_param[tuning_param["param_set"]]

        new_controller = FBLinearizingController(self.spiral_model, tuning, self._ros_node)

        with self.initialization_lock:
            self.controller = new_controller
            self.assign_trajectory_to_controller(start_time=self._time_since_traj_start()) 

    def _init_spiraling_MPC(self):
        tuning_param = self.params["tuning"]["spiraling"]

        self.spiral_params = {
            'horizon': self.params["horizon"],
            'param_set': tuning_param["param_set"],
            'tuning': tuning_param
        }

        new_controller = FancyMPC(self.spiral_model, self.spiral_params, self.robot_params, self._ros_node, 
                                  include_omega=True)

        with self.initialization_lock:
            self.controller = new_controller
            self.assign_trajectory_to_controller(start_time=self._time_since_traj_start()) 

    def _init_nominal_MPC(self):
        tuning_param = self.params["tuning"]["nominal"]

        self.nominal_params = {
            'horizon': self.params["horizon"],
            # "uub": [1] * self.model.m, # TODO
            "ulb": [0] * self.model.m,
            # "terminal_constraint": get_terminal_constraints_no_faults(self.model, tuning), # TODO
            'param_set': tuning_param["param_set"],
            'tuning': tuning_param
        }

        new_controller = GenericMPC(self.model, self.nominal_params, self._ros_node)

        with self.initialization_lock:
            self.controller = new_controller
            self.assign_trajectory_to_controller(start_time=self._time_since_traj_start()) 

    def add_actuator_failures(self, failures):
        # Add the faults to the model and failure list
        for fault in failures:
            fault = ensure_proper_fault_information(fault, self.model)
            # add to model
            self.actuator_failures.append(fault)
            self.model.add_actuator_fault([fault.pos1, fault.pos2], fault.intensity)

            self._ros_node.get_logger().info(f'Reactive controller registered actuator failure: ' +
                                f'[{fault.pos1}, {fault.pos2}], intensity: {fault.intensity}')

        # Update parameters
        self.input_bounds = InputBounds(self.model)
        self.input_handler = InputHandlerImproved(self.model, self.input_bounds)

        if self.input_handler.get_fault_category() != States.ORIGIN_IN_INTERIOR_OF_U:
            self.spiral_model = SpiralDynamics.from_ff_model(self.model)

        # Update controller if there already is one
        if self.controller is not None:
            self._set_controller()

    def _time_since_traj_start(self):
        """ Get the time difference in sec since the trajectory was started """
        if self.traj_start_time is None:
            self.traj_start_time = self._ros_node.get_clock().now()

        return (self._ros_node.get_clock().now() - self.traj_start_time).nanoseconds * 1e-9

    def get_control(self, x0, t):
        return self.controller.get_control(x0, t)

    def assign_trajectory_to_controller(self, start_time=0):
        """ Assign the trajectory to the (new) controller """
        if self.trajectory is None:
            self._ros_node.get_logger().info("No trajectory loaded yet - cannot assign to controller.")
            return

        start_idx = int(start_time / self.model.dt)

        print(f"Assigning trajectory to controller from index {start_idx}")
        print(f"Trajectory shape: {self.trajectory.shape}")

        self.controller.assign_trajectory(self.trajectory[:,start_idx:])
        self._ros_node.get_logger().info("Assigned trajectory to controller.")

    def load_trajectory(self, action, duration=10, file_path=None):
        """ Automatically assign the trajectory to the current controller & save trajectory """
        self.controller.load_trajectory(action, duration, file_path)
        self.trajectory = self.controller.get_trajectory(start_time=0)

    def get_trajectory(self):
        raise NotImplementedError

    def __repr__(self):
        return f"ReactiveController using ({self.controller})"

"""
- [ ] Needs to copy values s.a. trajectory
- [ ] What are the standard launch file arguments? 


- Need to have 
    - self.mode
    - self.time_step
    - traj_shape
    - traj_duration
    - actuator_failures
    "- trajectory_tracking"
- Need not to have
    - horizon
    - param_set
    - solver_opts
    - tuning

"""
