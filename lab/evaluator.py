import sys
import neat
import numpy as np
from simulator import Simulator
from gene import Gene
import tf_neat.visualize as visualize
from datetime import datetime
import os

class Evaluator :
    def __init__(self, make_net, make_env, param=None):

        self.batch_size=1

        # accept env, net
        self.envs = [make_env() for _ in range(self.batch_size)]
        self.make_net = make_net

        # accept system parameter
        self.param = param


    def eval_genome(self, genome, config, idx, debug=False):

        s = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
        rootPath = f"log\{s}"

        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        savePath = f"{rootPath}\{str(idx)}"
        os.makedirs(savePath)


        net = self.make_net(genome, config, self.batch_size)
        gene = Gene(net, self.param)
        gene.expression()
        model = gene.model

        #print
        #gene.print_model_info()
        gene.model_display(savePath)
        visualize.draw_net(config, genome, True, node_names={}, filename=f"{savePath}\\net.png")

        #simulate
        simulator = Simulator(model)
        input_vector = np.array([1,])
        simulator.input(input_vector)
        simulator.run(10)
        output_vector = simulator.output()
        print(f"output_vector {output_vector}")
        simulator.show()
        simulator.save(savePath)


        sys.exit()

        self.batch_size=1
        fitnesses = np.zeros(self.batch_size)
        states = [env.reset() for env in self.envs]
        dones = [False] * self.batch_size

        step_num = 0
        while True:
            step_num += 1
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            if debug:
                actions = self.activate_net(
                    net["f_r"], states, debug=True, step_num=step_num)
            else:
                actions = self.activate_net(net["f_r"], states)
            assert len(actions) == len(self.envs)

            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:
                    state, reward, done, _ = env.step(action)
                    fitnesses[i] += reward
                    if not done:
                        states[i] = state
                    dones[i] = done
            if all(dones):
                break

        return sum(fitnesses) / len(fitnesses)



