import numpy as np
import yaml
import copy
import warnings
from ament_index_python.packages import get_package_share_directory
import os

from micro_orbiting_mpc.models.ff_input_bounds import InputBounds, InputHandlerImproved
from micro_orbiting_mpc.util.utils import Rot3

class ControllerBaseClass:
    def __init__(self, ros_node):
        self.trajectory = None
        self.model = None
        self._ros_node = ros_node
        self.logger = self._ros_node.get_logger()

        # Publish ControllerValues messages. Publisher instantiated in the ROS node.
        self.controller_stats_pub = self._ros_node.controller_stats_pub

    def set_model(self, model):
        self.model = copy.deepcopy(model)
        self.dynamics = self.model.dynamics
        self.bounds = InputBounds(self.model)
        self.ih = InputHandlerImproved(self.model, self.bounds)

        self.mass = model.mass
        self.J = model.J
        self.dt = model.dt
        self.Nx, self.Nu = model.n, model.m

    def add_actuator_fault(self, fault_pos, fault_value):
        self.model.add_actuator_fault(fault_pos, fault_value)
        self.set_model(self.model) 

    def get_control(self, x0, t):
        """
        Takes the current time and state and returns the control input to apply to the system.
        """
        raise NotImplementedError("Method not implemented")

    def assign_trajectory(self, traj):
        raise NotImplementedError("Method not implemented")

    def load_trajectory(self, action, duration=100, file_path=None):
        """
        Load a trajectory for the controller to track.

        :param action: action to perform, either 'generate', 'generate_line' or 'load'
        :type action: str
        :param duration: duration of the trajectory, defaults to 'None'. This means the whole
                        trajectory should be loaded independent of the duration.
        :type duration: int, optional
        :param file_path: path to the trajectory file, defaults to None
        :type file_path: str, optional
        """
        def load_trajectory_from_file(file_path):
            """
            Load a trajectory from a yaml file.

            :param file_path: path to the file
            :type file_path: str
            """
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            if data['dt'] != self.dt:
                raise ValueError(f"Trajectory ({data['dt']}s) and controller ({self.dt}s) " + 
                                "have different time steps.")

            return np.array(data['x']).T

        def generate_trajectory(duration, form, **kwargs):
            """
            Get a trajectory for the robot to follow.
            """
            t = np.arange(0, 10*duration, self.dt )
            match form:
                case 'sin':
                    gain = 0.1
                    x_ref = np.stack((
                                gain*np.sin(t), 
                                t, 
                                t, 
                                gain*np.cos(t), 
                                np.ones(t.shape), 
                                np.ones(t.shape)
                            ))
                    
                case 'line':
                    x_ref = np.stack((
                                t, np.zeros(t.shape), np.zeros(t.shape), 
                                np.ones(t.shape), np.zeros(t.shape), np.zeros(t.shape)
                            ))
                case 'point_stabilizing' | 'hover':
                    if 'position' in kwargs:
                        x_pos = kwargs['position']
                    else:
                        x_pos = [0, 0, 0]
                    # x_ref = np.zeros((6, t.shape[0]))
                    x_ref = np.stack((
                        x_pos[0] * np.ones(t.shape),
                        x_pos[1] * np.ones(t.shape),
                        x_pos[2] * np.ones(t.shape),
                        np.zeros(t.shape),
                        np.zeros(t.shape),
                        np.zeros(t.shape)
                    ))
                case 'polynomial':
                    gain = 0.5
                    px = [-1.97286583226378e-15, 1.66180981216056e-13, 6.12318911042380e-11, -1.32571002201637e-08, 1.16484944065588e-06, -5.66119580093472e-05, 0.00163003172607062, -0.0276296048987181, 0.257558753061823, -1.11192499745334, 1.51531002733896, 7.20774354733917e-10]
                    py = [3.42808808505018e-08, -6.12608028153846e-06, 0.000402635893116014, -0.0116734662688749, 0.133886381426178, -0.224203064624999, -1.06756190411843e-12]
                    px = [pi * gain for pi in px]
                    py = [pi * gain for pi in py]
                    if duration > 60:
                        warnings.warn("Polynomial trajectory only gives reasonable results until 60s.")
                    x_ref = np.stack((
                        np.polyval(px, t),
                        np.polyval(py, t),
                        t,
                        np.polyval([pi * (len(px)-i) for i, pi in enumerate(px[:-1])], t),
                        np.polyval([pi * (len(py)-i) for i, pi in enumerate(py[:-1])], t),
                        np.ones_like(t)
                    ))
                case 'polynomial_old':
                    scale = 0.1
                    x_ref = np.stack((
                        scale * t * (scale * t-0.6) * (scale * t-2),
                        -0.5 * scale * t * (0.5*scale * t - 2),
                        np.zeros_like(t),
                        3 * scale**3 * t**2 -5.2 * scale**2 * t + scale * 1.2,
                        -0.5 * scale**2 * t + scale,
                        np.zeros_like(t)
                    ))
                case 'circle':
                    radius = kwargs['radius'] if 'radius' in kwargs else 2
                    sPerRot = kwargs['sPerFullCircle'] if 'sPerFullCircle' in kwargs else 30
                    # full turn every <x>s
                    omega = 2*np.pi/sPerRot 
                    x_ref = np.stack((
                        radius * np.cos(omega * t),
                        radius * np.sin(omega * t),
                        np.zeros_like(t),
                        -radius * omega * np.sin(omega * t),
                        radius * omega * np.cos(omega * t),
                        np.zeros_like(t)
                    ))

                    # turn the circle by 90°
                    R = Rot3(np.pi/2)
                    for i in range(x_ref.shape[1]):
                        x_ref[:3, i] = R @ x_ref[:3, i]
                        x_ref[3:, i] = R @ x_ref[3:, i]

                    # x_ref += np.array([[-radius], [0], [0], [0], [0], [0]])
                    x_ref += np.array([[1.75], [0], [0], [0], [0], [0]])
                    # import matplotlib.pyplot as plt
                    # plt.plot(x_ref[0], x_ref[1])
                    # plt.show()
                    # exit()
                case _:
                    raise ValueError("Invalid form. Use 'sin' or 'line'.")
            return x_ref

        if action == "generate_sin":
            t = generate_trajectory(duration, form='sin')
        elif action == "generate_line":
            t = generate_trajectory(duration, form='line')
        elif action == "generate_polynomial":
            t = generate_trajectory(duration, form='polynomial')
        elif action == "generate_point_stabilizing" or action == "hover":
            t = generate_trajectory(duration, form='point_stabilizing')
        elif 'hover' in action:
            name, *params = action.split('_')
            if name != 'hover':
                raise ValueError(f"Invalid action '{action}'.")
            if len(params) != 3:
                raise ValueError(f"Invalid number of parameters for action '{action}'. Use 'hover'"
                                 +" or 'hover_<x>_<y>_<alpha>'")
            position = [float(p) for p in params]
            t = generate_trajectory(duration, form='point_stabilizing', position=position)
        elif action == "generate_circle":
            t = generate_trajectory(duration, form='circle')
        elif 'circle' in action:
            name, *params = action.split('_')
            if name != 'circle':
                raise ValueError(f"Invalid action '{action}'.")
            if len(params) != 4:
                raise ValueError(f"Invalid number of parameters for action '{action}'. Use 'circle_r_<radius>_sPerFullCircle_<speed>'")
            if params[0] != 'r' or params[2] != 'sPerFullCircle':
                raise ValueError(f"Invalid parameters for action '{action}'. Use 'circle_r_<radius>_sPerFullCircle<speed>'")
            radius = float(params[1])
            speed = float(params[3])
            t = generate_trajectory(duration, form='circle', radius=radius, sPerFullCircle=speed)
        elif action == "load":
            fpath = file_path if file_path is not None else './controllers/trajectories/traj_01.yaml'
            t = load_trajectory_from_file(fpath) 
            if duration is not None:
                if t.shape[1] < (duration)/self.dt:
                    raise ValueError(f"Warning: Trajectory is too short: Has only lenth of  " +
                                     f"{t.shape[1]*self.dt}s, but {duration}s needed.")
        elif 'load' in action:
            name, f_name = action.split('=')
            config_dir = get_package_share_directory('micro_orbiting_mpc')
            path = os.path.join(config_dir, 'config', 'trajectories', f_name)

            if not os.path.exists(path):
                raise FileNotFoundError(f"File '{path}' not found.")

            if name != 'load':
                raise ValueError(f"Invalid action '{action}'.")

            t = load_trajectory_from_file(path)
        else:
            raise ValueError(f"Invalid action '{action}'.")
        
        self.assign_trajectory(t)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def get_trajectory(self, stop_time=None, start_time=0):
        if self.trajectory is None:
            return None

        if stop_time is None:
            try:
                return self.trajectory[:, int(start_time/self.dt):]
            except IndexError:
                self.logger.error(f"Trajectory too short: requested was the time from {start_time}"+ 
                                  f"s to end, but trajectory has only length of " +
                                  f" {self.trajectory.shape[1]*self.dt}s.")
        else:
            try:
                return self.trajectory[:, int(start_time/self.dt):int(stop_time/self.dt)]
            except IndexError:
                self.logger.error(f"Trajectory too short: requested was the time from {start_time}"+ 
                                  f"s to {stop_time}s, but trajectory has only length of " +
                                  f" {self.trajectory.shape[1]*self.dt}s.")

