import pickle
import sys
import threading
import _thread as thread

import numpy as np
import pandas as pd
from system import System
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    # print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f'Success. {end - start} taken for {func.__name__}')
        return result
    return wrapper

class Simulator :
    def __init__(self, model, engine="phy", ode_language="FORTRAN"):
        self.model = dict()
        self.model_dict = model
        ## pandas -> numpy
        for v in model:
            if type(model[v]) == pd.core.frame.DataFrame :
                self.model[v] = model[v].to_numpy()
            else :
                self.model[v] = model[v]

        ## simulate condition
        self.checksum = self.check_model_condition_fail()
        #self.checksum=True
        if self.checksum == True :
            self.system= 0
            self.history = {
                "age" :0,
                "xp" : list(),
                "t":list(),
            }

            self.engine = engine
            self.ode_language = ode_language
            self.method = 'RK23'  # 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
            self.system = System(self.model, engine=self.engine)

    @classmethod
    def by_file(cls, savepath):
        with open(f"{savepath}\model.JSON", 'rb') as fr:
            model = pickle.load(fr)
            return cls(model)


    def input(self, input_vector):
        input_vector = np.pad(input_vector, (0, self.model["neuron_num"] - len(input_vector)), 'constant', constant_values=0)
        input_vector = np.array([input_vector]).T
        self.model["x_0"] += np.dot(self.model["input_maker"], input_vector)

    def output(self):
        return np.dot(np.sum(self.history["xp"], axis=0).reshape(2,-1)[0], self.model["input_maker"])[-self.model["output_num"]:]

    def check_model_condition_fail(self):
        ## simulate condition

        if self.model["n"] == 0: return False
        if self.model["e"] == 0 : return False
        if self.model["e"] >= 100: return False

        return True

    #@exit_after(10)
    #@timed
    def simulating(self, xp0, t, ode_language, method=False) :
        ## select ode solver & solve
        if ode_language == "FORTRAN":
            ode_function = lambda xp, t: self.system.ode(t, xp)
            return odeint(ode_function, xp0, t)
        elif ode_language == "PYTHON":
            solution = solve_ivp(self.system.ode, [0, t[-1]], xp0, method=method, dense_output=True)
            return solution.sol(t).T


    def run(self, time):

        ## setting initial of ode
        xp0 = np.array([self.model["x_0"], self.model["p_0"]]).reshape(-1)
        t = np.linspace(0, time)

        ## simulate
        try :  xp = self.simulating( xp0, t, self.ode_language, method=self.method)
        except KeyboardInterrupt: return False

        ## result save
        xp_end = xp[-1,:].reshape(2,-1)
        self.model["x_0"], self.model["p_0"] = np.array([xp_end[0]]).T, np.array([xp_end[1]]).T
        self.history["xp"] = xp
        self.history["t"] = t

        # show or save result
        cmap = plt.cm.get_cmap('hsv', self.model["n"])
        for i,node_name in enumerate(self.model["node_names"]) :
            plt.plot(self.history["t"] + self.history['age'], self.history["xp"][:, i], color=cmap(i), label=node_name)
        # plt.ylim([0, 100])
        plt.xlabel('time')
        plt.legend(loc='best')


        self.history["age"] += time
        return True

    def visualize(self, savePath=False):
        if savePath :
            try:
                plt.savefig(f"{savePath}/model_{str(self.history['age'])}.png", dpi=300)
            except :
                pass
            with open(f"{savePath}\model.JSON", 'wb') as fw:
                pickle.dump(self.model_dict, fw)
        else : plt.show()





