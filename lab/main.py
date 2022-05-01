import os
import gym
import neat
from evaluator import Evaluator
from datetime import datetime
from tf_neat.recurrent_net import RecurrentNet
from tf_neat.population import Population
from tf_neat.parallel import ParallelEvaluator
from tf_neat.neat_reporter import LogReporter
import tensorflow as tf

# Activate eager TensorFlow execution
tf.executing_eagerly()

param = {
    "g_c": 2,
    "g_s": 3,
    "max_state": 5,
    "react_depth": 2,
    "neuron_num": 6,
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

def neat_cfg_change() :
    with open("neat_cfg_maker.cfg","r") as rnc :
        neat_cfg_data = rnc.read()
        num_inputs = 3*param["g_c"] + 2 + 2*param["g_s"]
        num_hidden = 12
        num_outputs = 13

        neat_cfg_data = neat_cfg_data.replace("num_inputs_replace", str(num_inputs))
        neat_cfg_data = neat_cfg_data.replace("num_hidden_replace", str(num_hidden))
        neat_cfg_data = neat_cfg_data.replace("num_outputs_replace", str(num_outputs))

    with open("neat.cfg","w") as wnc:
        wnc.write(neat_cfg_data)

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



    pop = Population(config)
    pe = ParallelEvaluator(1, evaluator.eval_genome,  rootPath=rootPath)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("./logs/neat.json", evaluator.eval_genome)
    pop.add_reporter(logger)
    pop.run(pe.evaluate, n_generations)


if __name__ == "__main__" :
    neat_cfg_change()
    run(100)