#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from micro_orbiting_msgs.msg import ControllerValues

class RealTimeVisualizer(Node):
    def __init__(self):
        super().__init__('real_time_visualizer')

        # Visualization parameters
        self.robot_width = 0.6
        self.robot_height = 0.6
        self.plot_limits = [-5, 5, -5, 5]  # [xmin, xmax, ymin, ymax]
        
        # Initialize state variables
        self.position = np.zeros(2)  # x1, y1
        self.orientation = 0.0  # alpha

        # Create subscription
        self.controller_sub = self.create_subscription(
            ControllerValues,
            '/px4_mpc/controller_values',
            self.controller_callback,
            10)

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

        # Create timer for visualization updates
        self.update_timer = self.create_timer(0.05, self.update_plot)  # 20Hz update rate

    def controller_callback(self, msg):
        self.position = np.array([msg.x1, msg.y1])
        self.orientation = msg.alpha

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

        # Trigger redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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