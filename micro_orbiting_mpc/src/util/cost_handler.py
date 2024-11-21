import sqlite3
import sympy as sp
import json
import numpy as np
import casadi as ca

from util.polytope import MyPolytope
from util.yes_no_question import query_yes_no

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
    def __init__(self) -> None:
        self.con = sqlite3.connect('controllers/explicitMPC_and_tuning/cost_function.db')
        self.cur = self.con.cursor()

    def create_table(self):
        # self.cur.execute("CREATE TABLE cost(F11 FLOAT, F12 FLOAT, F21 FLOAT, F22 FLOAT, F31 FLOAT, F32 FLOAT, F41 FLOAT, F42 FLOAT)")
        # self.cur.execute(''' CREATE TABLE cost(
        #                         F11 FLOAT,
        #                         F12 FLOAT,
        #                         F21 FLOAT,
        #                         F22 FLOAT,
        #                         F31 FLOAT,
        #                         F32 FLOAT,
        #                         F41 FLOAT,
        #                         F42 FLOAT,
        #                         cost STRING,
        #                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        #                     )''')
        if True:
        # if query_yes_no("Are you sure you want to delete and recreate the whole database?"):
            self.cur.execute("DROP TABLE IF EXISTS cost")
            self.cur.execute(''' CREATE TABLE cost(
                                    name STRING,
                                    t_cost STRING,
                                    t_set STRING,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                )''')
            self.con.commit()

    def get_cost_fcn_maybe_for_later(self, model):
        """
        Determine F11, F12, F21, F22, F31, F32, F41, F42 from the model
        Get the newest cost function from the database
        """
        F11 = 1
        F12 = 0
        F21 = 0
        F22 = 1
        F31 = 0
        F32 = 0
        F41 = 0
        F42 = 0

        self.cur.execute("SELECT * FROM cost ORDER BY created_at DESC LIMIT 1")
        cost = self.cur.fetchone()
        
        if cost:
            cost = eval(cost[0][0], {
                'sp':sp,
                'Symbol':sp.Symbol,
                'Float':sp.Float,
                'e0_1':sp.Symbol('e0_1'),
                'e0_3':sp.Symbol('e0_3'),
                'e0_2':sp.Symbol('e0_2'),
                'e0_4':sp.Symbol('e0_4'),
                'e0_5':sp.Symbol('e0_5')
            })
            return cost
        else:
            raise ValueError("No cost function found in database for the given model")

    def get_cost_fcn(self, model):
        res = self.cur.execute("SELECT t_cost, t_set FROM cost WHERE name = ? ORDER BY created_at DESC", ("cost1",))
        res = res.fetchall() # I think fetchone would instead not give me a list, but just the tuple with the requested values
        print(res[0][0].replace("Float('", "").replace("', precision=53)", ""))
        # print(res[0][0])
        terminal_cost = eval(res[0][0], {
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
        terminal_set = res[0][1]
        terminal_set = json.loads(terminal_set, cls=PolytopeDecoder)

        return terminal_cost, terminal_set

    def set_cost_fcn(self, tcost, tset, model):
        tset = json.dumps(tset, cls=PolytopeEncoder)
        self.cur.execute("INSERT INTO cost (name, t_cost, t_set) VALUES (?, ?, ?)", ("cost1", tcost, tset))
        self.con.commit()

if __name__ == "__main__":
    ch = CostHandler()
    # ch.create_table()
    ch.get_cost_fcn(None)
