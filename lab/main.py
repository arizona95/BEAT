import os
import gym
import neat
from evaluator import Evaluator
from datetime import datetime
from tf_neat.recurrent_net import RecurrentNet
from tf_neat.population import Population
from tf_neat.neat_reporter import LogReporter
import tensorflow as tf
# Activate eager TensorFlow execution
print("Executing eagerly: ", tf.executing_eagerly())

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

    def eval_genomes(genomes, config, now_generations):

        genRootPath = f"{rootPath}\gen_{now_generations}"
        try: os.mkdirs(genRootPath)
        except: pass
        for idx, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config, idx, rootPath=genRootPath)
            print(f"fitness : {genome.fitness}, key: {idx}")

    pop = Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("./logs/neat.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    pop = Population(config)
    pop.run(eval_genomes, n_generations)


if __name__ == "__main__" :
    run(100)