import yaml
import numpy as np

def read_yaml(file_path, *keys):
    """
    Reads a value from a YAML file. The keys should be passed ordered by the hierarchy level
    of the file from top- to low-level.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        result = data
        for key in keys:
            result = result[key]
        
        return result
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except KeyError as e:
        print(f"Error: Key '{e.args[0]}' not found in the YAML structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

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