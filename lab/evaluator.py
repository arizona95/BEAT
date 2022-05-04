import os
import sys
import pickle
import numpy as np
from gene import Gene
from simulator import Simulator
import tf_neat.visualize as visualize
import matplotlib.pyplot as plt

class Evaluator :
    def __init__(self, make_net, make_env, param=None):

        # accept env, net
        self.env = make_env()
        self.make_net = make_net

        # accept system parameter
        self.param = param
        self.engine = "phy"# bio, phy
        self.ode_language = 'FORTRAN'  # FORTRAN, PYTHON
        self.fail_fitness = -1


    def eval_genome(self, genome, config, idx=False, rootPath=False, debug=False):


        plt.clf()
        if rootPath:
        # make idx save folder
            savePath = f"{rootPath}\{str(idx)}"
            os.makedirs(savePath)

            with open(f"{savePath}\genome.JSON", 'wb') as fw:
                pickle.dump(genome, fw)

        net = self.make_net(genome, config, 1)
        gene = Gene(net, self.param)
        gene.expression()
        model = gene.model


        # simulator make
        simulator = Simulator(model, engine = self.engine, ode_language=self.ode_language)
        if simulator.checksum == False : return self.fail_fitness

        if rootPath:
            gene.model_display(savePath)
            visualize.draw_net(config, genome, True, node_names={}, filename=f"{savePath}\\net.png")


        # environment setting
        fitnesses = np.zeros(1)
        self.env.reset()
        input_vector = np.array([0]*self.param["input_num"])

        '''
        ## preparing
        success = simulator.run(1)
        if success == False:  return -1

        ##training
        for tr_num in range (10) :
            for epoch in range (300) :

                simulator.input(input_vector)
                simulator.run(0.1)
                output_vector = simulator.output()
                action = 1 if output_vector > 100 else 0
                state, reward, done, _ = self.env.step(action)
                input_vector = np.exp(state)
                plt.clf()

                if done: break 
        '''


        for epoch in range (300) :

            simulator.input(input_vector)
            success = simulator.run(0.1)
            if success == False:  return self.fail_fitness
            output_vector = simulator.output()
            action = 1 if output_vector > 100 else 0
            state, reward, done, _ = self.env.step(action)
            input_vector = np.minimum(1,np.exp(state))
            fitnesses += reward

            if rootPath and (done or (epoch+1)%10 == 0) :
                simulator.visualize(savePath=savePath)
                plt.clf()

            if done: break

        #print(f"fitnesses : {fitnesses[0]} , {idx}")
        return fitnesses[0]



