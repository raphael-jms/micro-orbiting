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

from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.controllers.spiralMPC_linearizing.spiral_mpc_v1 import SpiralMPC
from micro_orbiting_mpc.controllers.spiralMPC_eMPC.controller_empc import FancyMPC

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

        # get parameters
        self.mode = self.declare_parameter('mode', 'dummy').value

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

        # create publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_direct_actuator = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.predicted_path_pub = self.create_publisher(Path, '/px4_mpc/predicted_path', 10)
        self.reference_pub = self.create_publisher(Marker, "/px4_mpc/reference", 10)
        self.reference_pub = self.create_publisher(Marker, "/px4_mpc/reference", 10)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Create model and controller
        match (self.mode):
            case 'nominal' | 'reactive' :
                # nominal:  Assumes nominal case without actuator failures, no way to react
                # reactive: Assumes actuator failures can occur, but start without any
                self.model = FreeFlyerDynamicsFull(timer_period)
                self.controller = DummyModel()
            case 'spiralMPC_linearizing':
                # Actuator failures present from start, fb linearizing MPC controller
                self.model = DummyModel()
                self.controller = SpiralMPC(self.model)
            case 'spiralMPC_eMPC':
                # Actuator failures present from start, MPC controller based on eMPC
                self.model = DummyModel()
                self.controller = FancyMPC(self.model)
            case 'dummy':
                # Dummy controller for testing
                self.model = DummyModel()
                self.controller = DummyController()
            case _:
                self.get_logger().error('Unknown mode: ' + self.mode)
                return

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.orientation_x, self.orientation_y, self.orientation_z = 0.0, 0.0, 0.0
        self.angular_velocity_x, self.angular_velocity_y, self.angular_velocity_z = 0.0, 0.0, 0.0
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])

        self.updater = GiveUpdate(timer_period, rate=1)

    def vehicle_local_position_callback(self, msg):
        """
        Callback for local position
        Transform directly from NED (North, East, Down) to standard x-y-z coordinate system
        """
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

        self.orientation_z = -msg.heading

    def vehicle_attitude_callback(self, msg):
        """
        Callback for local orientation
        Transform from quaternion to angle and then from NED (North, East, Down) to standard x-y-z coordinate system
        """
        heading = quat2euler(msg.q, axes="szyx") # Both message and library with quaternion form (w, x, y, z)
                                                 # Transform: static (rotation around global axes) z->y->x
        self.orientation_z = - heading[0]
        self.orientation_y = heading[2]
        self.orientation_x = heading[1]

    def vehicle_angular_velocity_callback(self, msg):
        """
        Callback for local angular velocity
        Transform directly from NED (North, East, Down) to standard x-y-z coordinate system
        """
        self.angular_velocity_x = msg.xyz[1]
        self.angular_velocity_y = msg.xyz[0]
        self.angular_velocity_z = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print(f"NAV_STATUS: {msg.nav_state} - offboard status: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}")
        self.nav_state = msg.nav_state

    def publish_control(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE:
        # Output is float[16]
        
        thrust_controller = u_pred.flatten() / self.model.max_force  # normalizes w.r.t. max thrust
        """
        thruster order in controller: [F11, F12, F21, F22, F31, F32, F41, F42]
                               index:  0    1    2    3    4    5    6    7

        Alignment of the thruster definitions between the controller and the simulation environment
        where the numbers below are the indices of the simulation input signal
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
        thrust_simulator[0] = thrust_controller[6]
        thrust_simulator[1] = thrust_controller[7]
        thrust_simulator[2] = thrust_controller[4]
        thrust_simulator[3] = thrust_controller[5]
        thrust_simulator[4] = thrust_controller[2]
        thrust_simulator[5] = thrust_controller[3]
        thrust_simulator[6] = thrust_controller[0]
        thrust_simulator[7] = thrust_controller[1]

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

        x0 = np.array([error_position[0],
                        error_position[1],
                        self.vehicle_local_velocity[0],
                        self.vehicle_local_velocity[1],
                        self.orientation_z,
                        self.angular_velocity_z]).reshape(-1, 1)

        u_pred, u_simple = self.controller.get_control(x0, 0.0)
        # self.updater.update(f"Pos: {self.vehicle_local_position} \t Control: {actuator_outputs_msg.control} \t OffboardMode: {self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD}")
        self.updater.update(f"Pos: {self.vehicle_local_position} \t Attitude: {self.orientation_z} \t Velocity: {self.vehicle_local_velocity} \t Angular Velocity: {self.angular_velocity_z} \n \t Control {u_simple}, actual control: {u_pred}")

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
