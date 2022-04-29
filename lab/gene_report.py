import os
import neat
import pickle
import numpy as np
from gene import Gene
from tf_neat.recurrent_net import RecurrentNet
np.set_printoptions(precision=6, suppress=True)

def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)

param = {
    "g_c":2,
    "g_s":2,
    "max_state":5,
    "react_depth":1,
    "neuron_num":5,
    "input_num":4,
    "output_num":1
}

config_path = os.path.join(os.path.dirname(os.path.abspath('')), "lab/neat.cfg")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

time = "2022-04-29-15h-15m-46s"
gen_idx =1
idx = 3
filepath = f"generations\\{time}\\gen_{gen_idx}\\{idx}"

genome = ""
with open(f"{filepath}\genome.JSON", 'rb') as fr:
    genome = pickle.load(fr)

net = make_net(genome, config, 1)
gene = Gene(net, param)
gene.expression()
model = gene.model