#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from micro_orbiting_msgs.msg import ControllerValues
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pprint

class TrajectoryVisualizer(Node):
    """
    ROS2 Node that visualizes the trajectory of the center point as well as the planned trajectory 
    for the horizon.
    """
    def __init__(self):
        super().__init__('trajectory_visualizer')
        
        time_recorded = 10 # seconds
        self.sec_between_plans = 2 # show plan every _ seconds
        self.dt = 0.1
        
        data_length = int(time_recorded/self.dt)
        plan_data_length = int(time_recorded/self.sec_between_plans)
        self.plan_data_length = plan_data_length
        # Initialize time and data storage
        self.times = deque(maxlen=data_length)
        self.data = {
            'x': deque(maxlen=data_length),
            'y': deque(maxlen=data_length),
            'omega': deque(maxlen=data_length),
            'vx': deque(maxlen=data_length),
            'vy': deque(maxlen=data_length),
            'plan_x': deque(maxlen=plan_data_length),
            'plan_y': deque(maxlen=plan_data_length),
            'plan_omega': deque(maxlen=plan_data_length),
            'plan_vx': deque(maxlen=plan_data_length),
            'plan_vy': deque(maxlen=plan_data_length)
        }

        self.fig, self.axes = plt.subplots(5, 1, figsize=(10, 12))
        self.setup_plots()
        
        self.subscription = self.create_subscription(
            ControllerValues,
            '/px4_mpc/controller_values',
            self.callback,
            10)

        self.timer = self.create_timer(0.1, self.update_plots)
        self.record_plan_counter=0

    def setup_plots(self):
        titles = [
            'X Position', 
            'Y Position', 
            'X Velocity', 
            'Y Velocity',
            'Angular Velocity'
        ]
        self.lines = {}
        
        for ax, title in zip(self.axes, titles):
            ax.set_title(title)
            ax.grid(True)
            self.lines[title] = {
                'actual': ax.plot([], [], 'b-', label='Actual')[0],
            }
            for i in range(self.plan_data_length):
                self.lines[title][f"plan{i}"] = ax.plot([], [], 'r--', label='Planned')[0]
            # ax.legend()

        plt.tight_layout()

    def callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.times.append(current_time)
        self.data['x'].append(msg.center_state_x)
        self.data['y'].append(msg.center_state_y)
        self.data['omega'].append(msg.center_state_omega)
        self.data['vx'].append(msg.center_state_vx)
        self.data['vy'].append(msg.center_state_vy)

        if self.record_plan_counter%(self.sec_between_plans/self.dt) == 0:
            self.horizon = len(msg.plan_x1)
            times = np.arange(current_time, current_time+self.dt*self.horizon, self.dt)
            self.data['plan_x'].append({"plan": msg.plan_x1, "times":times})
            self.data['plan_y'].append({"plan": msg.plan_y1, "times":times})
            self.data['plan_omega'].append({"plan": msg.plan_omega, "times":times})
            self.data['plan_vx'].append({"plan": msg.plan_x2, "times":times})
            self.data['plan_vy'].append({"plan": msg.plan_y2, "times":times})
        
        self.record_plan_counter += 1

    def update_plots(self):
        if not self.times:
            return

        times_array = np.array(self.times)
        
        data_mapping = {
            'X Position': 'x',
            'Y Position': 'y',
            'Angular Velocity': 'omega', 
            'X Velocity': 'vx',
            'Y Velocity': 'vy'
        }

        for title, key in data_mapping.items():
            self.lines[title]['actual'].set_data(times_array, np.array(self.data[key]))
            for i, cur_plan in enumerate(self.data[f"plan_{key}"]):
                self.lines[title][f'plan{i}'].set_data(
                    np.array(cur_plan["times"]),
                    np.array(cur_plan["plan"])
                )

        for ax in self.axes:
            ax.set_xlim([self.times[0], self.times[-1]+self.horizon*self.dt])
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main():
    rclpy.init()
    visualizer = TrajectoryVisualizer()

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