#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Raphael Stöckner"
__contact__ = "stockner@kth.se"
# This code is strongly inspired by https://github.com/Jaeyoung-Lim/px4-mpc/tree/master

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from transforms3d.euler import quat2euler
import math
import time
from ament_index_python.packages import get_package_share_directory
import os
import casadi as ca

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import ActuatorMotors

from micro_orbiting_msgs.srv import SetPose
from micro_orbiting_msgs.srv import SetActuatorFailure
from micro_orbiting_msgs.msg import SetTrajectory
from micro_orbiting_msgs.msg import ControllerValues

from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull, \
    FreeFlyerDynamicsSimplified, SpiralDynamics
from micro_orbiting_mpc.controllers.spiralMPC_linearizing.spiral_mpc_v1 import SpiralMPC
from micro_orbiting_mpc.controllers.spiralMPC_eMPC.controller_empc import FancyMPC
from micro_orbiting_mpc.controllers.controller_mpc_base import GenericMPC
from micro_orbiting_mpc.controllers.nominalMPC_no_faults.terminal_constraints_no_faults import get_terminal_constraints_no_faults
from micro_orbiting_mpc.controllers.fb_linearizing_controller import FBLinearizingController
from micro_orbiting_mpc.models.ff_input_bounds import SpiralParameters
from micro_orbiting_mpc.util.utils import read_yaml, read_ros_parameter_file

from micro_orbiting_mpc.test.dummy import DummyModel, DummyController

def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = position[0]
    pose_msg.pose.position.y = position[1]
    pose_msg.pose.position.z = position[2]
    pose_msg.pose.orientation.x = attitude[0]
    pose_msg.pose.orientation.y = attitude[1]
    pose_msg.pose.orientation.z = attitude[2]
    pose_msg.pose.orientation.w = attitude[3]
    return pose_msg

class GiveUpdate:
    """
    Prints an update to the console every rate seconds
    """
    def __init__(self, dt, rate=1):
        self.no_calls = int(rate/dt) # number of calls until message is shown
        self.counter = 0
    
    def update(self, msg):
        self.counter += 1
        if self.counter % self.no_calls == 0:
            self.counter = 0
            print(msg)

class SpacecraftMPCNode(Node):
    """
    ROS2 node for controlling the spacecraft using MPC
    """
    def __init__(self):
        super().__init__('spacecraft_mpc_node')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Define default parameters
        robot_params = {
            'mode': 'dummy',
            'time_step': 0.1,
            'horizon': 10,
            'trajectory_tracking': True,
            'param_set': "P1",
            'solver_opts': { 'ipopt': {'tol': 1e-5} },
            'traj_shape': "generate_point_stabilizing",
            'traj_duration': 30,
            'tuning': {
                'P1': { 'Q': [1.0] * 6, 'R': [1.0] * 3, 'R_full': [1.0] * 8, 'P_mult': 1.0 },
            },
            'actuator_failures': []
        }

        # Helper function to flatten and declare parameters 
        # (ROS2 doesn't allow nested parameters, instead they are declared as tuning.P1.Q etc.)
        def declare_params_recursive(params, prefix=''):
            for key, value in params.items():
                if key == 'tuning' or key == 'actuator_failures':
                    continue # Load these seperately
                param_name = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    declare_params_recursive(value, f"{param_name}.")
                else:
                    self.declare_parameter(param_name, value)

        declare_params_recursive(robot_params)

        # Get the actual parameter values (either from the parameter server or the default values)
        def get_params_recursive(params, prefix=''):
            result = {}
            for key, value in params.items():
                if key == 'tuning' or key == 'actuator_failures':
                    continue # Load these seperately
                param_name = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    result[key] = get_params_recursive(value, f"{param_name}.")
                else:
                    result[key] = self.get_parameter(param_name).value
            return result

        actual_params = get_params_recursive(robot_params)

        # Set class attributes using the actual values
        for param, value in actual_params.items():
            setattr(self, param, value)

        robot_parameter_defaults = {
            "mass": 14.5, "inertia": 0.370, "max_force": 1.75, "thruster_distance_to_center": 0.14
        }
        self.robot_parameters = {}
        for param in robot_parameter_defaults.keys():
            self.declare_parameter(param, robot_parameter_defaults[param])
            self.robot_parameters[param] = self.get_parameter(param).value

        self.declare_parameter('config_file', 'nominal_mpc.yaml')
        config_file = self.get_parameter('config_file').value
        self.tuning = read_ros_parameter_file(config_file, 'tuning')
        self.actuator_failures = read_ros_parameter_file(config_file, 'actuator_failures')

        # create publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode, 
            '/fmu/in/offboard_control_mode', 
            qos_profile)
        self.publisher_direct_actuator = self.create_publisher(
            ActuatorMotors, 
            '/micro_orbiting/control_signal',
            # '/fmu/in/actuator_motors', 
            qos_profile)
        self.predicted_path_pub = self.create_publisher(
            Path, 
            '/px4_mpc/predicted_path', 
            10)
        self.reference_pub = self.create_publisher(
            Marker, 
            "/px4_mpc/reference", 
            10)
        self.controller_stats_pub = self.create_publisher(
            ControllerValues,
            "/px4_mpc/controller_values",
            10)

        # Create model and controller
        match (self.mode):
            case 'faultfree' | 'reactive' :
                # faultfree:  Assumes nominal case without actuator failures, no way to react
                # reactive: Assumes actuator failures can occur, but starts without any
                self.model = FreeFlyerDynamicsFull(self.time_step, self.robot_parameters)
                self.params = {
                    "horizon": self.horizon,
                    "uub": [1] * self.model.m, # as inputs are normalized
                    "ulb": [0] * self.model.m,
                    "tuning": self.tuning,
                    "param_set": self.param_set,
                    "terminal_constraint": get_terminal_constraints_no_faults(self.model, 
                                                                    self.tuning[self.param_set]),
                }
                self.controller = GenericMPC(self.model, self.params, self)
            case 'spiralMPC_linearizing':
                # Actuator failures present from start, MPC controller based on linear MPC
                # initialize SpiralModel
                self.model = self.initialize_damaged_spiral_model()
                self.declare_parameter("recalculate_terminal_set", False)

                self.params = {
                    "horizon": self.horizon,
                    "tuning": self.tuning,
                    "param_set": self.param_set,
                    "recalculate_terminal_set": self.get_parameter("recalculate_terminal_set").value
                }

                self.controller = SpiralMPC(self.model, self.params, self)
            case 'spiralMPC_eMPC':
                # Actuator failures present from start, MPC controller based on eMPC
                self.model = self.initialize_damaged_spiral_model()
                self.params = {
                    "horizon": self.horizon,
                    "tuning": self.tuning,
                    "param_set": self.param_set,
                }
                self.controller = FancyMPC(self.model, self.params, self)
            case 'feedback_linearizing_controller':
                # initialize SpiralModel
                self.model = self.initialize_damaged_spiral_model()
                self.controller = FBLinearizingController(self.model, self.tuning[self.param_set], self)
            case 'dummy':
                # Dummy controller for testing
                self.model = DummyModel()
                self.controller = DummyController(self)
            case _:
                self.get_logger().error('Unknown mode: ' + self.mode)
                return

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.orientation_x, self.orientation_y, self.orientation_z = 0.0, 0.0, 0.0
        self.angular_velocity_x, self.angular_velocity_y, self.angular_velocity_z = 0.0, 0.0, 0.0
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        # create subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile)

        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile)

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)

        self.trajectory_sub = self.create_subscription(
            SetTrajectory,
            'trajectory_commands',
            self.trajectory_callback,
            qos_profile)

        # create services
        self.actuator_failure_sub = self.create_service(
            SetActuatorFailure,
            'actuator_failure',
            self.actuator_failure_callback)

        # wait_for_initial_trajectory() returns False if no trajectory is received within 5 seconds
        if not self.wait_for_initial_trajectory():
            self.get_logger().info('Defaulting to hovering as no other trajectory arrived yet.')
            default_traj = SetTrajectory()
            default_traj.action = "hover"
            default_traj.duration = 100
            self.trajectory_callback(default_traj)

        self.timer = self.create_timer(self.time_step, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.updater = GiveUpdate(self.time_step, rate=1)

    def wait_for_initial_trajectory(self, timeout_sec=5.0):
        """ Wait for the first trajectory to arrive, abort after timeout_sec and return False """
        start_time = self.get_clock().now()
        
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.controller.trajectory is not None:
                return True
        return False

    def initialize_damaged_spiral_model(self):
        """
        Initialize the model and controller for the damaged case
        Returns SpiralDynamics model
        """
        # Actuator failures present from start, fb linearizing MPC controller
        self.sim_model = FreeFlyerDynamicsFull(self.time_step, self.robot_parameters)

        # Add actuator failures to the model
        for fault in self.actuator_failures:
            self.sim_model.add_actuator_fault(fault["act_ids"], fault["intensity"])

        # Used model is SpiralDynamics. With the creation of this models, the spiral
        # parameters are already fixed. 
        return SpiralDynamics.from_ff_model(self.sim_model)

    def actuator_failure_callback(self, request, response):
        """
        Callback for actuator failure service
        """
        response.success = False

        if self.mode != 'reactive':
            self.get_logger().warn("Adding errors during runtime is only supported for mode " +
                                   f"'reactive'. Current mode is '{self.mode}'.")
            return response

        for idx in range(len(request.actuators)):
            # Validate input
            if not 0 <= request.intensity[idx] <= 1:
                self.get_logger().warn('Invalid intensity value')
                return response

            if not len(request.actuators[idx]) == 2:
                self.get_logger().warn('Invalid number of actuators')
                return response

            # Add actuator failure
            try:
                self.controller.add_actuator_fault(request.actuators[idx], request.intensity[idx])
                self.get_logger().info(f'Registered actuator failure: [{request.actuators}], ' +
                                    f'intensity: {request.intensity}')
            except Exception as e:
                self.get_logger().warn(f'Failed to set actuator failure: {str(e)}')
                return response

        response.success = True
        return response

    def vehicle_attitude_callback(self, msg):
        """
        Callback for attitude
        Transform from NED (North, East, Down) to non-standard NWU (North, West, Up) coordinate sys
        """
        new_quaternion = [msg.q[0], msg.q[1], -msg.q[2], -msg.q[3]]

        heading = quat2euler(new_quaternion, axes="szyx") # Both message and library with quaternion form (w, x, y, z)
                                                 # Transform: static (rotation around global axes) z->y->x
        self.orientation_z = heading[0]
        self.orientation_y = heading[2]
        self.orientation_x = heading[1]

    def vehicle_local_position_callback(self, msg):
        """
        Callback for local position
        Transform from NED (North, East, Down) to non-standard NWU (North, West, Up) coordinate sys
        """
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

        self.orientation_z = -msg.heading

    def vehicle_angular_velocity_callback(self, msg):
        """
        Callback for angular velocity
        Transform from NED (North, East, Down) to non-standard NWU (North, West, Up) coordinate sys
        """
        self.angular_velocity_x = msg.xyz[0]
        self.angular_velocity_y = -msg.xyz[1]
        self.angular_velocity_z = -msg.xyz[2]

    # def vehicle_local_position_callback(self, msg):
    #     """
    #     Callback for local position
    #     Transform directly from NED (North, East, Down) to standard ENU coordinate system
    #     """
    #     self.vehicle_local_position[0] = msg.y
    #     self.vehicle_local_position[1] = msg.x
    #     self.vehicle_local_position[2] = -msg.z
    #     self.vehicle_local_velocity[0] = msg.vy
    #     self.vehicle_local_velocity[1] = msg.vx
    #     self.vehicle_local_velocity[2] = -msg.vz

    #     self.orientation_z = -msg.heading

    # def vehicle_attitude_callback(self, msg):
    #     """
    #     Callback for local orientation
    #     Transform from quaternion to angle and then from NED (North, East, Down) to standard ENU coordinate system
    #     """
    #     heading = quat2euler(msg.q, axes="szyx") # Both message and library with quaternion form (w, x, y, z)
    #                                              # Transform: static (rotation around global axes) z->y->x
    #     self.orientation_z = - heading[0]
    #     self.orientation_y = heading[2]
    #     self.orientation_x = heading[1]

    # def vehicle_angular_velocity_callback(self, msg):
    #     """
    #     Callback for local angular velocity
    #     Transform directly from NED (North, East, Down) to standard ENU coordinate system
    #     """
    #     self.angular_velocity_x = msg.xyz[1]
    #     self.angular_velocity_y = msg.xyz[0]
    #     self.angular_velocity_z = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print(f"NAV_STATUS: {msg.nav_state} - offboard status: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}")
        self.nav_state = msg.nav_state

    def trajectory_callback(self, msg):
        try:
            self.controller.load_trajectory(
                action=msg.action,
                duration=msg.duration,
                file_path=msg.file_path if msg.file_path else None
            )
            self.get_logger().info(f'Loaded new trajectory: {msg.action}')
        except ValueError as e:
            self.get_logger().error(f'Failed to load trajectory: {str(e)}')
        except AttributeError as e:
            self.get_logger().error(f'No controller initialized: {str(e)}')

    def publish_control(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE: Output is float[16]
        
        if isinstance(u_pred, ca.DM):
            u_pred = u_pred.full().flatten() # Convert casadi to numpy

        thrust_controller = u_pred.flatten() / self.model.max_force  # normalizes w.r.t. max thrust
        """
        thruster order in controller: [F11, F12, F21, F22, F31, F32, F41, F42]
                               index:  0    1    2    3    4    5    6    7

        Alignment of the thruster definitions between the controller and the simulation environment
        where the numbers below are the indices of the simulation input signal

        Drawing is according to the axes visible in Gazebo.
                4     6
                ▲     ▲
                │F21  │F11
             ┌──┴─────┴──┐
        2 ◄──┤           ├──► 3
          F31│     ▲     │F32
             │     │x    │
             │  ◄──┘     │
          F41│    y      │F42
        0 ◄──┤           ├──► 1
             └──┬─────┬──┘
                │F22  │F12
                ▼     ▼
                5     7
        """
        thrust_simulator = np.zeros(12, dtype=np.float32)
        # thrust_simulator[0] = thrust_controller[6]
        # thrust_simulator[1] = thrust_controller[7]
        # thrust_simulator[2] = thrust_controller[4]
        # thrust_simulator[3] = thrust_controller[5]
        # thrust_simulator[4] = thrust_controller[2]
        # thrust_simulator[5] = thrust_controller[3]
        # thrust_simulator[6] = thrust_controller[0]
        # thrust_simulator[7] = thrust_controller[1]
        """
        The coordinate transforms to the NWU system lead to an angle information that is turned by
        90 degree (i.e. the local coordinate system does not align with the one shown in Gazebo).
        Thus, the actuators are turned once more:
                0     2
                ▲     ▲
                │F21  │F11
             ┌──┴─────┴──┐
        5 ◄──┤           ├──► 4
          F31│  _  ▲_    │F32
             │  y  │x    │
             │  ◄──┘     │
          F41│           │F42
        7 ◄──┤           ├──► 6
             └──┬─────┬──┘
                │F22  │F12
                ▼     ▼
                1     3
        """
        thrust_simulator[0] = thrust_controller[2]
        thrust_simulator[1] = thrust_controller[3]
        thrust_simulator[2] = thrust_controller[0]
        thrust_simulator[3] = thrust_controller[1]
        thrust_simulator[4] = thrust_controller[5]
        thrust_simulator[5] = thrust_controller[4]
        thrust_simulator[6] = thrust_controller[7]
        thrust_simulator[7] = thrust_controller[6]

        actuator_outputs_msg.control = thrust_simulator.flatten()
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.thrust_and_torque = False
        offboard_msg.direct_actuator = True
        self.publisher_offboard_mode.publish(offboard_msg)

        self.setpoint_position = np.zeros_like(self.vehicle_local_position)
        error_position = self.vehicle_local_position - self.setpoint_position

        x0 = np.array([ 
                        error_position[0],
                        error_position[1],
                        self.orientation_z,
                        self.vehicle_local_velocity[0],
                        self.vehicle_local_velocity[1],
                        self.angular_velocity_z]).reshape(-1, 1)

        u_pred = self.controller.get_control(x0, 0.0)
        # self.updater.update(f"Pos: {self.vehicle_local_position} \t Control: {actuator_outputs_msg.control} \t OffboardMode: {self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD}")
        self.updater.update(f"Pos: {self.vehicle_local_position} \t Attitude: {self.orientation_z} \t Velocity: {self.vehicle_local_velocity} \t Angular Velocity: {self.angular_velocity_z} actual control: {u_pred}")

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_control(u_pred)

def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = SpacecraftMPCNode()

    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
