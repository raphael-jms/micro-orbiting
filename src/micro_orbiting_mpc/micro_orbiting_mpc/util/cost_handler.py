import sqlite3
import sympy as sp
import json
import numpy as np
import casadi as ca
import os
import pprint

from micro_orbiting_mpc.util.polytope import MyPolytope
from micro_orbiting_mpc.util.yes_no_question import query_yes_no

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class PolytopeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MyPolytope):
            return {
                'A': obj.A.tolist(),
                'b': obj.b.tolist()
            }
        return super().default(obj)

class PolytopeDecoder(json.JSONDecoder):
    def decode(self, s):
        obj = json.loads(s)
        if 'A' in obj and 'b' in obj:
            return MyPolytope(np.array(obj['A']), np.array(obj['b']))
        return obj

class CostHandler:
    """
    Class that saves and loads the cost function of the spiraling MPC controller. Uses SQLite
    under the hood because saving text files could (would) be a mess.
    """
    def __init__(self, db_name = 'spiralMPC_empc_cost.db') -> None:
        # Create cache dir if it does not exist
        cache_dir = os.path.join(os.path.expanduser('~'), '.ros', 'cache', 'micro_orbiting_mpc')
        os.makedirs(cache_dir, exist_ok=True)

        # Get the file location
        file_location = os.path.join(cache_dir, db_name)
        self.db_name = db_name
        self.file_location = file_location

        # Connect to the database
        self.con = sqlite3.connect(file_location)
        self.cur = self.con.cursor()

        def table_exists(name):
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            return self.cur.fetchone() is not None
        
        if not table_exists('cost'):
            raise ValueError(f"Table 'cost' does not exist in the database {file_location}. " + \
                             "Please run the setup according to the README.md file.")

    def __del__(self):
        if hasattr(self, 'con'):  
            self.con.close()

    def create_table(self, overwrite=False):
        if overwrite or \
            query_yes_no("Are you sure you want to delete and recreate the whole database?"):

            self.cur.execute("DROP TABLE IF EXISTS cost")

            self.cur.execute(''' CREATE TABLE cost(
                                    F11 FLOAT,
                                    F12 FLOAT,
                                    F21 FLOAT,
                                    F22 FLOAT,
                                    F31 FLOAT,
                                    F32 FLOAT,
                                    F41 FLOAT,
                                    F42 FLOAT,
                                    tuning STRING,
                                    robot_params STRING,
                                    t_cost STRING,
                                    t_set STRING,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                )''')
            self.con.commit()

    def get_cost_fcn(self, model, tuning, robot_params):
        """
        Get the newest cost function from the database
        """
        # self.cur.execute("SELECT * FROM cost ORDER BY created_at DESC LIMIT 1")
        self.cur.execute('''
            SELECT t_cost, t_set 
            FROM cost 
            WHERE F11 = ? AND F12 = ? AND F21 = ? AND F22 = ? 
                AND F31 = ? AND F32 = ? AND F41 = ? AND F42 = ?
                AND tuning = ? AND robot_params = ?                         
            ORDER BY created_at DESC 
            LIMIT 1
        ''', self.get_db_lookup_params(model, tuning, robot_params))

        result = self.cur.fetchone()
        
        if result:
            terminal_cost = eval(result[0], {
                'sp':sp,
                'Symbol':sp.Symbol,
                'Float':sp.Float,
                'Abs':sp.Abs,
                'tanh':sp.tanh,
                'ca':ca,
                'e0_1':sp.Symbol('e0_1'),
                'e0_3':sp.Symbol('e0_3'),
                'e0_2':sp.Symbol('e0_2'),
                'e0_4':sp.Symbol('e0_4'),
                'e0_5':sp.Symbol('e0_5')
                })
            terminal_set = result[1]
            terminal_set = json.loads(terminal_set, cls=PolytopeDecoder)

            return terminal_cost, terminal_set
        else:
            error_msg = f"No cost function found in database at {self.file_location} for the " + \
                        "given model. You requested terminal ingredients for settings\n" + \
                        pprint.pformat(tuning) + "\n" + \
                        pprint.pformat(robot_params) + "\n" + \
                        pprint.pformat(model.failed_actuators) + "\n" + \
                        "Please run the setup first according to the README.md file."
            raise ValueError(error_msg)

    def set_cost_fcn(self, tcost, tset, model, tuning, robot_params):
        tset = json.dumps(tset, cls=PolytopeEncoder)
        self.cur.execute('''
            INSERT INTO cost(F11, F12, F21, F22, F31, F32, F41, F42, tuning, robot_params, 
                         t_cost, t_set)
                         VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                         (*self.get_db_lookup_params(model, tuning, robot_params), str(tcost), tset))
        self.con.commit()

    def get_db_lookup_params(self, model, tuning, robot_params):
        """ 
        Determine the parameters F11, F12, F21, F22, F31, F32, F41, F42 for the database lookup 
        from the model.
        """
        f8 = [-1.0]*8 # -1 if it has not failed
        for actuator in model.failed_actuators:

            # Convert to float if necessary
            intensity = actuator["intensity"] 
            if isinstance(intensity, np.ndarray):
                intensity = intensity.item()

            f8[actuator["idx"]] = intensity

        f8.append(json.dumps(tuning, sort_keys=True))
        f8.append(json.dumps(robot_params, sort_keys=True))

        return f8

if __name__ == "__main__":
    ch = CostHandler()
    # ch.create_table()
    ch.get_cost_fcn(None)
