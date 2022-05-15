import os
import gym
import neat
from evaluator import Evaluator
from datetime import datetime
from tf_neat.recurrent_net import RecurrentNet
from tf_neat.adaptive_linear_net import AdaptiveLinearNet
from tf_neat.adaptive_net import AdaptiveNet
from tf_neat.population import Population
from tf_neat.parallel import ParallelEvaluator
from tf_neat.neat_reporter import LogReporter
from tf_neat.checkpoint import Checkpointer
import tensorflow as tf
rootPath =""

def make_env():
    return gym.make("CartPole-v1")

def make_net(genome, config, bs):
    #input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    #output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    return RecurrentNet.create(
        genome,
        config,
        #input_coords=input_coords,
        #output_coords=output_coords,
        batch_size=bs
    )

def neat_cfg_change() :
    with open("neat_cfg_maker.cfg","r") as rnc :
        neat_cfg_data = rnc.read()
        num_inputs = 3*param["g_c"] + 2 + 2*param["g_s"]
        num_hidden = 14
        num_outputs = 14

        neat_cfg_data = neat_cfg_data.replace("num_inputs_replace", str(num_inputs))
        neat_cfg_data = neat_cfg_data.replace("num_hidden_replace", str(num_hidden))
        neat_cfg_data = neat_cfg_data.replace("num_outputs_replace", str(num_outputs))

    with open("neat.cfg","w") as wnc:
        wnc.write(neat_cfg_data)

def run(n_generations, checkpoint_path=False):
    global rootPath

    if checkpoint_path != False :
        pop = Checkpointer.restore_checkpoint(checkpoint_path)
    else :

        config_path = os.path.join(os.path.dirname(os.path.abspath('')), "lab\\neat.cfg")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        pop = Population(config)

    evaluator = Evaluator(
        make_net, make_env=make_env, param=param
    )

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("./logs/neat.json", evaluator.eval_genome)
    pop.add_reporter(logger)
    pop.add_reporter(Checkpointer(1, filename_prefix =rootPath))
    pe = ParallelEvaluator(3, evaluator.eval_genome, rootPath=rootPath)
    pop.run(pe.evaluate, n_generations)


if __name__ == "__main__" :
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


    n_generations = 30
    load_generation = 4
    load_rootpath = "2022-05-09-07h-18m-45s"
    filename = f"generations\\{load_rootpath}\\gen_{load_generation}\\genomes"

    now_time = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
    rootPath = f"generations\\{now_time}"
    if not os.path.exists(rootPath):   os.makedirs(rootPath)
    neat_cfg_change()


    preTraining= False
    #preTraining = True
    if preTraining == False : run(n_generations)
    elif preTraining == True : run(n_generations, checkpoint_path =filename)