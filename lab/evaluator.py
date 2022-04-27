import sys
import numpy as np
from simulator import Simulator
from gene import Gene
import tf_neat.visualize as visualize
import os
import matplotlib.pyplot as plt

class Evaluator :
    def __init__(self, make_net, make_env, param=None):

        # accept env, net
        self.env = make_env()
        self.make_net = make_net

        # accept system parameter
        self.param = param


    def eval_genome(self, genome, config, idx, rootPath, debug=False):

        plt.clf()
        # make idx save folder
        savePath = f"{rootPath}\{str(idx)}"
        os.makedirs(savePath)

        with open(f"{savePath}\genome.JSON", 'wb') as fw:
            pickle.dump(genome, fw)

        net = self.make_net(genome, config, 1)
        gene = Gene(net, self.param)
        gene.expression()
        model = gene.model


        # if show
        #savePath=False

        #print
        #gene.print_model_info()
        gene.model_display(savePath)
        if savePath:
            visualize.draw_net(config, genome, True, node_names={}, filename=f"{savePath}\\net.png")

        # simulator make
        simulator = Simulator(model)

        # environment setting

        fitnesses = np.zeros(1)
        states = self.env.reset()
        dones = False

        input_vector = np.array([0]*self.param["input_num"])

        for epoch in range (300) :

            #print(f"age : {simulator.history['age']}")
            #print(f"input_vector  :{input_vector}")
            simulator.input(input_vector)
            success = simulator.run(0.1)
            if success == False :  return -1
            output_vector = simulator.output()
            #print(f"output_vector  :{output_vector}")
            action = 1 if output_vector > 100 else 0
            #print(f"action : {action}")

            state, reward, done, _ = self.env.step(action)
            input_vector = np.exp(state)
            fitnesses += reward

            if done or (epoch+1)%10 == 0 :
                simulator.visualize(savePath=savePath)
                #simulator.visualize()  # show

            if done: break



        #sys.exit()
        return fitnesses[0]



