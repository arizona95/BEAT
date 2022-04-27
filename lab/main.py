import os
import gym
import neat
from evaluator import Evaluator
from datetime import datetime
from tf_neat.recurrent_net import RecurrentNet

param = {
    "g_s": 3,
    "s_s": 2,
    "max_state": 5,
    "react_depth": 1,
    "neuron_num": 5,
    "input_num": 4,
    "output_num": 1
}

s = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
rootPath = f"generations\{s}"

if not os.path.exists(rootPath):
    os.makedirs(rootPath)


def make_env():
    return gym.make("CartPole-v0")


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def run(n_generations):
    config_path = os.path.join(os.path.dirname(os.path.abspath('')), "lab/neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = Evaluator(
        make_net, make_env=make_env, param=param
    )

    def eval_genomes(genomes, config):
        for idx, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config, idx, rootPath=rootPath)

    pop = neat.Population(config)
    pop.run(eval_genomes, n_generations)


if __name__ == "__main__" :
    run(10)