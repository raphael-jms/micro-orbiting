#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque

from micro_orbiting_msgs.msg import ControllerValues, FailedActuators
from micro_orbiting_mpc.util.utils import Rot3, Rot3Inv, Rot, RotInv

class RealTimeVisualizer(Node):
    """
    Node to visualize the real-time state of the spacecraft and the controller inputs.
    In contrast to gazebo, no physical visualization is provided, but an abstracted view enriched
    with the current controller inputs and potentially the orbit center.
    """
    def __init__(self):
        super().__init__('real_time_visualizer')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Visualization parameters
        self.robot_width = 0.6
        self.robot_height = 0.6
        self.force_scaler = 0.15
        self.plot_limits = [-5, 5, -5, 5]  # [xmin, xmax, ymin, ymax]
        self.max_force = 1.75
        
        # Calculate number of points to store for path
        path_duration = 30.0  # seconds
        update_rate = 0.1    # seconds (20Hz)
        self.path_points = int(path_duration / update_rate)
        self.disconnected_timeout = 5.0  # seconds

        # Initialize state variables
        self.position = np.zeros(2)  # x1, y1
        self.orientation = 0.0  # alpha
        self.angular_velocity = 0.0  # omega
        self.center_position = np.zeros(2)
        self.center_pos_full = np.zeros(5)  # x, y, vx, vy, omega
        self.resulting_force = np.zeros(3)
        self.desired_state = np.zeros(6)
        self.last_update_time = None

        # Initialize path storage
        self.robot_path = deque(maxlen=self.path_points)
        self.center_path = deque(maxlen=self.path_points)
        
        # Pre-fill paths with initial positions
        for _ in range(self.path_points):
            self.robot_path.append(np.zeros(2))
            self.center_path.append(np.zeros(2))

        # Create subscription
        self.controller_sub = self.create_subscription(
            ControllerValues,
            '/px4_mpc/controller_values',
            self.controller_callback,
            10)

        self.failed_forces_sub = self.create_subscription(
            FailedActuators,
            '/micro_orbiting/failed_actuators',
            self.failed_forces_callback,
            qos_profile
        )

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
        self.ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # Create visualization elements
        self.robot_rect = Rectangle(
            (0, 0),
            self.robot_width,
            self.robot_height,
            fill=False,
            linewidth=2,
            color='blue'
        )
        self.ax.add_patch(self.robot_rect)

        # Create orientation arrow
        self.orientation_arrow = self.ax.arrow(
            0, 0, 0, 0,
            head_width=0.1,
            head_length=0.2,
            fc='blue',
            ec='blue'
        )

        # Create center point
        self.center_point = Circle(
            (0, 0),
            radius=0.075,
            color='red',
            alpha=0.5,
            fill=True
        )
        self.ax.add_patch(self.center_point)

        # Desired state point
        self.desired_point = Circle(
            (0, 0),
            radius=0.03,
            color='black',
            fill=True
        )
        self.ax.add_patch(self.desired_point)

        # Create connection line
        self.connection_line, = self.ax.plot([], [], '--', color='gray')

        # Create path lines
        self.robot_path_line, = self.ax.plot([], [], '-', color='blue', alpha=0.5)
        self.center_path_line, = self.ax.plot([], [], '-', color='red', alpha=0.5)

        # Force visualization parameters
        self.force_arrows = []
        val1 = 0.6 * self.robot_width/2
        val2 = self.robot_width/2
        self.pos_orient = np.array([
            [ val2, -val1,  1,  0],
            [-val2, -val1, -1,  0],
            [ val2,  val1,  1,  0],
            [-val2,  val1, -1,  0],
            [ val1,  val2,  0,  1],
            [ val1, -val2,  0, -1],
            [-val1,  val2,  0,  1],
            [-val1, -val2,  0, -1],
        ])

        # Create force arrows
        for _ in range(8):
            arrow = self.ax.arrow(0, 0, 0, 0, 
                                head_width=0.05, 
                                head_length=0.1, 
                                fc='black', 
                                ec='black',
                                alpha=1.0)
            self.force_arrows.append(arrow)

        self.forces = np.zeros(8)  # Initialize forces array

        # Create failed forces
        self.failed_force_arrows = []

        for _ in range(8):
            arrow = self.ax.arrow(0, 0, 0, 0, 
                                head_width=0.05, 
                                head_length=0.1, 
                                fc='red', 
                                ec='red',
                                alpha=1.0)
            self.failed_force_arrows.append(arrow)

        self.failed_actuator_forces = np.zeros(8)

        # Create timer for visualization updates
        self.update_timer = self.create_timer(0.05, self.update_plot)  # 20Hz update rate
        self.forget_timer = self.create_timer(1.0, self.forget_if_disconnected)

    def controller_callback(self, msg):
        self.position = np.array([msg.x1, msg.y1])
        self.orientation = msg.alpha
        self.angular_velocity = msg.omega
        self.center_position = np.array([msg.center_state_x, msg.center_state_y])
        self.center_pos_full = np.array([msg.center_state_x, 
                                         msg.center_state_y, 
                                         msg.center_state_vx, 
                                         msg.center_state_vy,
                                         msg.center_state_omega])
        self.forces = np.array(msg.u_full)
        self.resulting_force = np.array(msg.u)
        
        # Update paths
        self.robot_path.append(self.position)
        self.center_path.append(self.center_position)
        self.desired_state = np.array(msg.desired_state)

        self.last_update_time = self.get_clock().now()
    
    def failed_forces_callback(self, msg):
        self.failed_actuator_forces = np.zeros(8)
        for failure in msg.failed_actuators:
            idx = failure.idx
            self.failed_actuator_forces[idx] = failure.intensity * self.max_force

    def update_plot(self):
        # Calculate rotated offsets for rectangle position
        R = np.array([[np.cos(self.orientation), -np.sin(self.orientation)],
                    [np.sin(self.orientation), np.cos(self.orientation)]])
        
        # Calculate bottom-left corner of rotated rectangle
        offset = R @ np.array([-self.robot_width/2, -self.robot_height/2])
        rect_x = self.position[0] + offset[0]
        rect_y = self.position[1] + offset[1]
        
        # Update rectangle
        self.robot_rect.set_xy((rect_x, rect_y))
        self.robot_rect.angle = np.degrees(self.orientation)

        # Update orientation arrow
        arrow_length = max(self.robot_width, self.robot_height) * 1.2
        dx = arrow_length * np.cos(self.orientation)
        dy = arrow_length * np.sin(self.orientation)
        self.orientation_arrow.set_data(
            x=self.position[0],
            y=self.position[1],
            dx=dx,
            dy=dy
        )

        # Update center point
        self.center_point.center = self.center_position
        self.desired_point.center = self.desired_state[:2]

        # Update connection line
        self.connection_line.set_data(
            [self.position[0], self.center_position[0]],
            [self.position[1], self.center_position[1]]
        )

        # Update force arrows
        self.force_arrows = self.update_forces(self.forces, self.force_arrows, R)
        self.failed_force_arrows = self.update_forces(self.failed_actuator_forces, self.failed_force_arrows, R)

        # Update paths
        robot_path_array = np.array(self.robot_path)
        center_path_array = np.array(self.center_path)
        
        self.robot_path_line.set_data(
            robot_path_array[:, 0],
            robot_path_array[:, 1]
        )
        self.center_path_line.set_data(
            center_path_array[:, 0],
            center_path_array[:, 1]
        )

        # Trigger redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_forces(self, forces, force_arrows, R):
        for i in range(8):
            if abs(forces[i]) < 1e-6:
                force_arrows[i].set_alpha(0.0)
                continue
            else:
                force_arrows[i].set_alpha(1.0)

            # Get local positions
            local_pos = self.pos_orient[i]
            force_magnitude = forces[i] * self.force_scaler  # Scale factor for visualization
            
            # Calculate start and end points
            start_point = np.array([local_pos[0], local_pos[1]])
            direction = np.array([local_pos[2], local_pos[3]])
            end_point = start_point + direction * force_magnitude

            # Transform to global coordinates
            start_global = self.position + R @ start_point
            end_global = self.position + R @ end_point

            # Update arrow
            dx = end_global[0] - start_global[0]
            dy = end_global[1] - start_global[1]
            force_arrows[i].set_data(x=start_global[0], 
                                        y=start_global[1],
                                        dx=dx, 
                                        dy=dy)
        return force_arrows

    def forget_if_disconnected(self):
        # clear the travelled path if the controller is disconnected
        if self.last_update_time is not None:
            time_diff = self.get_clock().now() - self.last_update_time
            if time_diff.nanoseconds * 1e-9 > self.disconnected_timeout:
                # Reset the past path
                for _ in range(self.path_points):
                    self.robot_path.append(np.zeros(2))
                    self.center_path.append(np.zeros(2))
                self.last_update_time = None

def main(args=None):
    rclpy.init(args=args)
    visualizer = RealTimeVisualizer()
    
    plt.show(block=False)
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()