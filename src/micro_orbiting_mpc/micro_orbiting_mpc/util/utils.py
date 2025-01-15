import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import matplotlib.pyplot as plt
import copy

def read_yaml(file_name, *keys):
    """
    Reads a value from a YAML file. The keys should be passed ordered by the hierarchy level
    of the file from top- to low-level.
    """
    try:
        config_dir = get_package_share_directory('micro_orbiting_mpc')
        config_file = os.path.join(config_dir, 'config', file_name)

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"File '{config_file}' not found.")

        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
        
        result = data

        for key in keys:
            result = result[key]
        
        return result
    except FileNotFoundError:
        print(f"Error: File '{config_file}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except KeyError as e:
        print(f"Error: Key '{e.args[0]}' not found in the YAML structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

def read_ros_parameter_file(file_name, *keys):
    return read_yaml(file_name, *["/**", "ros__parameters"], *keys)

def read_yaml_matrix(file_path, *keys):
    diagonal = read_yaml(file_path, *keys)
    return np.diag(diagonal)

class LogData:
    def __init__(self, dt, name):
        """
        Class containing the logging data

        :param dt: time step
        :param name: name of the data points as list
        """
        self.time = []
        self.data = []
        self.dt = dt
        self.name = name

    def add_data(self, time, data):
        """
        Add data to the log

        :param time: time
        :param data: data
        """
        self.time.append(time)
        if isinstance(data, np.ndarray):
            data = data.flatten()
        self.data.append(data)

    def get_array(self):
        """
        Returns the data as numpy array

        :return: numpy array
        """
        d = np.array(self.data).T
        t = np.array(self.time)

        if len(d.shape) == 3:
            if d.shape[0] == 1:
                d = d[0, :, :]
            elif d.shape[1] == 1:
                d = d[:, 0, :]
            elif d.shape[2] == 1:
                d = d[:, :, 0]
            else:
                print("Warning: Data has shape {d.shape}.")

        return d, t
        # return np.array(self.data).T, np.array(self.time)

    def len(self):
        """ Return number of data points """
        return len(self.time)

class EllipticalTerminalConstraint:
    def __init__(self, alpha, P):
        self.alpha = alpha
        self.P = P
    
    def __repr__(self):
        return f"EllipticalTerminalConstraint(alpha={self.alpha},\nP={self.P})"

    def plot_slice(self, dims, ax=None):
        """
        Plot a 2D slice through the ellipse where the value of the removed dimensions is set to zero
        
        input:
        dims: list of indices of rows/columns to be removed
        """
        # remove dimesions 
        P = np.delete(self.P.copy(), dims, axis=0)
        P = np.delete(P, dims, axis=1)

        if P.shape != (2,2):
            raise ValueError("The matrix P should be 2x2. Select appropriate dimensions to remove.")

        alpha = self.alpha

        if ax is None:
            fig, ax = plt.subplots(1,1)

        xx,yy = np.linspace(-1,1, 1000), np.linspace(-1,1, 1000)
        x,y = np.meshgrid(xx,yy)
        ax.contourf(x, y, (P[0,0]*x**2 + P[1,1]*y**2 + P[0,1]*x*y + P[1,0]*x*y), [0, alpha], cmap='Reds')
        ax.set_aspect('equal','datalim')

        return ax

def Rot(alplha):
    """
    Rotation matrix in 2D
    """
    return np.array([[np.cos(alplha), -np.sin(alplha)],
                     [np.sin(alplha), np.cos(alplha)]])

def Rot3(alpha):
    """
    Rotation matrix in 3D
    """
    return np.array([[np.cos(alpha), -np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])

def RotInv(alplha):
    """
    Inverse rotation matrix in 2D
    """
    return np.array([[np.cos(alplha), np.sin(alplha)],
                     [-np.sin(alplha), np.cos(alplha)]])

def Rot3Inv(alpha):
    """
    Inverse rotation matrix in 3D
    """
    return np.array([[np.cos(alpha), np.sin(alpha), 0],
                     [-np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q