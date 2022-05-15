import os
import gym
import neat
import pickle
import numpy as np
from gene import Gene
from simulator import Simulator
import matplotlib.pyplot as plt
from matplotlib import animation
from tf_neat.recurrent_net import RecurrentNet

np.set_printoptions(precision=10, suppress=True)


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def make_env():
    return gym.make("CartPole-v0")


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.clf()
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


param = {
    "g_c": 2,
    "g_s": 3,
    "max_state": 5,
    "react_depth": 2,
    "neuron_num": 6,
    "input_num": 4,
    "output_num": 1
}

config_path = os.path.join(os.path.dirname(os.path.abspath('')), "lab/neat.cfg")
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

filepath = f"test_case"
case_num = 2

savePath = f"{filepath}\\{case_num}"
genome = ""
with open(f"{savePath}\\genome.JSON", 'rb') as fr:
    genome = pickle.load(fr)

net = make_net(genome, config, 1)
gene = Gene(net, param)
gene.expression()
gene.model_display(savePath)
model = gene.model
simulator = Simulator(model, engine="bio", ode_language='FORTRAN')  # FORTRAN, PYTHON

env = make_env()
env.reset()
frames = []
input_vector = np.array([0] * param["input_num"])
fitnesses = np.zeros(1)
for epoch in range(300):
    frames.append(env.render(mode="rgb_array"))
    simulator.input(input_vector)
    success = simulator.run(1)
    if success == False:  done = True
    output_vector = simulator.output()
    action = 1 if output_vector[0] > 250 else 0
    # print(output_vector)
    state, reward, done, _ = env.step(action)
    # print(np.exp(state))
    input_vector = 10 * np.minimum(10, np.exp(state))
    fitnesses += reward

    if savePath and (done or (epoch + 1) % 10 == 0):
        simulator.visualize(savePath=savePath)
        plt.clf()

    if done: break

env.close()
save_frames_as_gif(frames, path=savePath + '\\')