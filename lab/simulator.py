import os
import sys
import pickle
import numpy as np
import pandas as pd
from system import System
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Simulator :
    def __init__(self, model):
        self.model = dict()
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
        print(self.model["output_num"])
        return np.dot(np.sum(self.history["xp"], axis=0).reshape(2,-1)[0], self.model["input_maker"])[-self.model["output_num"]:]

    def run(self, time):
        ## simulate condition
        if len(self.model["x_0"]) == 0: return False
        if len(self.model["M_"]) == 0: return False

        model = self.model

        model["n"] = model["x_0"].shape[0]
        model["c"] = model["m_c"].shape[0]
        model["e"] = model["M_"].shape[1]

        self.system = System(model)
        xp0 = np.array([model["x_0"], model["p_0"]]).reshape(-1)
        t = np.linspace(0, time)
        xp = odeint(self.system.ode, xp0, t)

        self.history["xp"] = xp
        self.history["t"] = t
        self.history["age"] += time

        return True


    def show(self, save=False, savePath=""):
        def get_cmap(n, name='hsv'):
            return plt.cm.get_cmap(name, n)

        node_names = self.model["node_names"]

        cmap = get_cmap(len(node_names))

        for i,node_name in enumerate(node_names) :
            plt.plot(self.history["t"], self.history["xp"][:, i], color=cmap(i), label=node_name)
        plt.xlabel('time')
        plt.legend(loc='best')


        if not save :
            plt.show()
        else :
            plt.savefig(f"{savePath}/model.png", dpi=300)

        '''
        energy = list()
        for i in range(len(xp)):
            energy.append(self.system.energy(np.array([self.history["xp"][i].reshape(2, -1)[0]]).T,\
                                             np.array([self.history["xp"][i].reshape(2, -1)[1]]).T, sim))

        plt.plot(self.history["t"], energy, 'm-', label='e')
        plt.xlabel('time')
        plt.legend(loc='best')
        if not save:
            plt.show()
        else:
            plt.savefig(f"{savePath}/energy.png", dpi=300)
        '''


    def save(self,savePath):

        with open(f"{savePath}\model.JSON", 'wb') as fw:
            pickle.dump(self.model, fw)

        self.show(save=True, savePath=savePath)




