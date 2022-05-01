import pickle
import numpy as np
import pandas as pd
from system import System
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Simulator :
    def __init__(self, model):
        self.model = dict()
        self.model_dict = model
        ## pandas -> numpy
        for v in model:
            if type(model[v]) == pd.core.frame.DataFrame :
                self.model[v] = model[v].to_numpy()
            else :
                self.model[v] = model[v]
        self.system= 0
        self.history = {
            "age" :0,
            "xp" : list(),
            "t":list(),
        }

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

    def run(self, time):
        ## simulate condition
        if self.model["x_0"].shape[0] == 0: return False
        if self.model["M_"].shape[1] > 100 : return False
        if self.model["M_"].shape[0] == 0: return False

        mode = 'bio'
        model = self.model
        if self.history["age"] ==0 :
            self.system = System(model, mode=mode)
        xp0 = np.array([model["x_0"], model["p_0"]]).reshape(-1)
        t = np.linspace(0, time)
        #xp = odeint(self.system.ode, xp0, t)
        method = 'RK23'  # available methods: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        #ode_function = lambda t, xp: self.system.ode(t, xp, mode)
        solution = solve_ivp(self.system.ode, [0, time], xp0, method=method, dense_output=True)
        xp = solution.sol(t).T

        xp_end = xp[-1,:].reshape(2,-1)

        self.model["x_0"], self.model["p_0"] = np.array([xp_end[0]]).T, np.array([xp_end[1]]).T

        self.history["xp"] = xp
        self.history["t"] = t


        # show or save result
        cmap = plt.cm.get_cmap('hsv', self.model["n"])
        for i,node_name in enumerate(self.model["node_names"]) :
            plt.plot(self.history["t"] + self.history['age'], self.history["xp"][:, i], color=cmap(i), label=node_name)
        #plt.ylim([0, 100])
        plt.xlabel('time')
        plt.legend(loc='best')


        self.history["age"] += time
        return True

    def visualize(self, savePath=False):
        if savePath :
            plt.savefig(f"{savePath}/model_{str(self.history['age'])}.png", dpi=300)
            with open(f"{savePath}\model.JSON", 'wb') as fw:
                pickle.dump(self.model_dict, fw)
        else : plt.show()





