import sys
import neat
import itertools
import numpy as np
import pandas as pd
from system import System
from scipy.integrate import odeint

import graphviz
import matplotlib.pyplot as plt
from IPython.display import display
import tf_neat.visualize as visualize

class Evaluator :
    def __init__(self, make_net, make_env, param=None):

        self.batch_size=1

        # accept env, net
        self.envs = [make_env() for _ in range(self.batch_size)]
        self.make_net = make_net

        # accept system parameter
        self.c_v = param["c_v"]
        self.s_v = param["s_v"]
        self.max_state = param["max_state"]

        # node,edge naming rules
        self.vector_separator = "/"
        self.gene_space_separator = "-"
        self.chemical_flow_character = "r"
        self.diffusion_flow_character = "d"
        self.hamiltonian_flow_character = "h"

    def eval_genome(self, genome, config, debug=False):

        net = self.make_net(genome, config, self.batch_size)

        def net_output( input_vector ) :
            return net.activate([np.array(input_vector)])[0]

        def to_string(li) :
            return self.vector_separator.join(str(e) for e in li)


        ## 1. Generation Gene Map G_SET

        c_0 = [0] * (self.c_v -1)
        g_0 = [0] * self.c_v
        s_0 = [0] * self.s_v


        m_c = list()
        q_c = list()
        for i in range(self.c_v-1):
            c_vector = [0]*(self.c_v-1)
            c_vector[i] =1
            input_vector = c_vector + g_0 + s_0 + g_0 + s_0
            reslut = net_output(input_vector)
            m_c.append(float(reslut[0]))
            q_c.append(2*float(reslut[1])-1)

        L=1

        rule_num = 1
        react_rules = dict()
        G0 = []
        for i in range(self.c_v-1):
            add_g0 = [0]*(self.c_v-1)+[0]
            add_g0[i]=1
            G0.append(add_g0)

        G=G0
        for i in range(L) :
            add_G = []
            for gi in G :
                for gj in G :
                    input_vector = c_0 + gi + s_0 + gj + s_0
                    react_or_not = 2*self.max_state*net_output(input_vector)[2]- self.max_state
                    react_rate = float(net_output(input_vector)[3])
                    if react_or_not >=0 :
                        new_g = list(np.array(gi[:-1])+np.array(gj[:-1]))+[int(react_or_not)]
                        add_G.append(new_g)
                        react_rules[self.chemical_flow_character+str(rule_num)] = {
                            "rule": [gi,gj, new_g],
                            "k" : react_rate,
                        }
                        rule_num +=1
            G=G+add_G

        ## make A dict

        A = dict()
        for i, gi in enumerate(G):
            input_vector = c_0 + gi + s_0 + g_0 + s_0
            A[to_string(gi)] = 5*float(net_output(input_vector)[3])


        ## 2. Generation Space Map S_SET

        S = []
        S_substrate = list(map(lambda x: list(x), list(itertools.product([0, 1], repeat=self.s_v))))
        for si in S_substrate :
            input_vector = c_0 + g_0 + si + g_0 + s_0
            space_or_not = 2*net_output(input_vector)[4]-1
            if space_or_not>=0 :
                S.append(si)

        ## make V matrix

        V = dict()
        for i, si in enumerate(S) :
            ## fs(si) = f(0,si,0,0)[3]
            input_vector = c_0 + g_0 + si + g_0 + s_0
            V[to_string(si)] = float(net_output(input_vector)[5])


        ## make D matrix

        D = [[0 for x in range(len(S))] for y in range(len(S))]
        for i, si in enumerate(S) :
            for j, sj in enumerate(S) :
                if i!=j:
                    input_vector = c_0 + g_0 + si + g_0 + sj
                    distance = 2*float(net_output(input_vector)[6])
                    if distance <1 : ## neighbor
                        D[i][j] = distance
                        D[j][i] = distance
                    else :
                        D[i][j] = np.inf
                        D[j][i] = np.inf


        ## 3. Generation real node
        node = dict()  # name:[x0, c, s, a, xh, h]


        def node_name_of(gi, si) :
            return to_string(gi)+ self.gene_space_separator +to_string(si)

        for i, gi in enumerate(G) :
            for j, sj in enumerate(S) :
                input_vector = c_0 + gi + sj + g_0 + s_0
                result = net_output(input_vector)
                x0 = 20*float(result[7])-10
                xh = 20*float(result[8])-10
                h = float(result[9])
                if xh <=0 :
                    h =0
                    xh=0
                if x0 >0 :
                    node_name = node_name_of(gi, sj)
                    node[node_name] = [
                        x0, #x0
                        gi[:-1],  # c
                        sj,  # s
                        A[to_string(gi)] * V[to_string(sj)],  # a
                        xh,  # xh
                        h,  # h
                        gi,
                    ]

        ## Generation Edge

        M_T = dict()
        k = dict()
        v = dict()

        ## chemical flow : same space
        space_node = dict()
        for i, si in enumerate(S):
            space_node[to_string(si)] = dict()

        for node_name in node :
            space_node[to_string(node[node_name][2])][to_string(node[node_name][6])] = node[node_name][6]

        for react_rule_name in react_rules :
            react_rule = react_rules[react_rule_name]["rule"]
            for space_name in space_node :
                if  to_string(react_rule[0]) in space_node[space_name] and \
                    to_string(react_rule[1]) in space_node[space_name] and \
                    to_string(react_rule[2]) in space_node[space_name] :

                    edge_name = react_rule_name + self.gene_space_separator + space_name

                    #add M_T
                    M_T_add = dict()
                    for node_name in node:
                        M_T_add[node_name] = 0
                    M_T_add[to_string(react_rule[0])+self.gene_space_separator+space_name] += 1
                    M_T_add[to_string(react_rule[1])+self.gene_space_separator+space_name] += 1
                    M_T_add[to_string(react_rule[2])+self.gene_space_separator+space_name] = -1
                    M_T[edge_name] = M_T_add

                    # add k
                    k[edge_name] = react_rules[react_rule_name]["k"]

                    # add v
                    v[edge_name] = 0

        ## diffusion flow, hamiltonian flow : different space
        for i, space_name_i in enumerate(space_node):
            si = list( map(lambda x:int(x), space_name_i.split(self.vector_separator) ) )
            for j, space_name_j in enumerate(space_node):
                sj = list(map(lambda x: int(x), space_name_j.split(self.vector_separator)))
                #if neighbor space
                if i!=j and D[i][j] > 0 and D[i][j] <1 :
                    for n , node_name_n in enumerate(space_node[space_name_i]) :
                        for m, node_name_m in enumerate(space_node[space_name_j]) :
                            if node_name_n == node_name_m :
                                input_vector = c_0 + space_node[space_name_i][node_name_n] +\
                                               si + space_node[space_name_j][node_name_m] + sj

                                ##print(f"input_vector: {input_vector}")
                                #diff edge
                                diff_k = 2*float(net_output(input_vector)[11])-1
                                if diff_k >0 : # edge connect

                                    edge_name = self.diffusion_flow_character + \
                                                self.gene_space_separator + node_name_n + \
                                                self.gene_space_separator + space_name_j + \
                                                self.gene_space_separator + space_name_i

                                    # add M_T
                                    M_T_add = dict()
                                    for node_name in node:
                                        M_T_add[node_name] = 0
                                    M_T_add[node_name_n + self.gene_space_separator + space_name_i] = 1
                                    M_T_add[node_name_m + self.gene_space_separator + space_name_j] = -1
                                    M_T[edge_name] = M_T_add

                                    # add k
                                    k[edge_name] = diff_k

                                    # add v
                                    v[edge_name] = 0

                                #hamilt edge
                                hamilt_k = 2 * float(net_output(input_vector)[12]) - 1
                                if hamilt_k>0 : # edge connect

                                    edge_name = self.hamiltonian_flow_character + \
                                                self.gene_space_separator + node_name_n + \
                                                self.gene_space_separator + space_name_j + \
                                                self.gene_space_separator + space_name_i

                                    # add M_T
                                    M_T_add = dict()
                                    for node_name in node:
                                        M_T_add[node_name] = 0
                                    M_T_add[node_name_n + self.gene_space_separator + space_name_i] = 1
                                    M_T_add[node_name_m + self.gene_space_separator + space_name_j] = -1
                                    M_T[edge_name] = M_T_add

                                    # add k
                                    k[edge_name] = 0

                                    # add v
                                    v[edge_name] = hamilt_k


        # make simulator entity
        sim = dict()
        sim["x_0"] = dict()
        sim["p_0"] = dict()
        sim["M"] = dict()
        sim["M_"] = dict()
        sim["S"] = dict()
        sim["D"] = dict()
        sim["a"] = dict()
        sim["k"] = dict()
        sim["v"] = dict()
        sim["c"] = dict()
        sim["x_h"] = dict()
        sim["h"] = dict()



        for node_name in node :
            sim["x_0"][node_name] = node[node_name][0]
            sim["p_0"][node_name] = 0
            sim["M"][node_name] = node[node_name][1]

            add_S = dict()
            for i, si in enumerate(S) :
                if to_string(si) == to_string(node[node_name][2]) :add_S[to_string(si)] =1
                else : add_S[to_string(si)] =0
            sim["S"][node_name] = add_S
            sim["a"][node_name] = node[node_name][3]
            sim["c"][node_name] = 1
            sim["x_h"][node_name] = node[node_name][4]
            sim["h"][node_name] = node[node_name][5]

        sim["x_0"] = pd.DataFrame.from_dict([sim["x_0"]]).T
        sim["p_0"] = pd.DataFrame.from_dict([sim["p_0"]]).T
        sim["M"] = pd.DataFrame.from_dict(sim["M"]).T
        sim["M_"] = pd.DataFrame.from_dict(M_T)
        sim["S"] = pd.DataFrame.from_dict(sim["S"]).T
        space_names = list(map(lambda x:to_string(x),S))
        sim["D"] = pd.DataFrame.from_dict(D)
        sim["D"].columns = space_names
        sim["D"] = sim["D"].T
        sim["D"].columns = space_names
        sim["m_c"] = pd.DataFrame.from_dict([m_c]).T
        sim["q_c"] = pd.DataFrame.from_dict([q_c]).T
        sim["a"] = pd.DataFrame.from_dict([sim["a"]]).T
        sim["k"] = pd.DataFrame.from_dict([k]).T
        sim["v"] = pd.DataFrame.from_dict([v]).T
        sim["c"] = pd.DataFrame.from_dict([sim["c"]]).T
        sim["x_h"] = pd.DataFrame.from_dict([sim["x_h"]]).T
        sim["h"] = pd.DataFrame.from_dict([sim["h"]]).T

        ### print, display

        '''
        print(f"react_rules : {react_rules}")
        print(f"G set : {G}")
        print(f"A set : {A}")
        print(f"S set: {S}")
        print(f"D mat: {D}")
        pprint.pprint(f"node list : {node}")

        for v in sim :
            print(v)
            display(sim[v])

        '''

        ## graph display

        display(sim["M_"])

        # gene graph
        node_names = {}
        stats = neat.StatisticsReporter()
        visualize.draw_net(config, genome, True, node_names=node_names)

        # sim graph

        g = graphviz.Graph('G', filename='../graph/test.gv', engine='fdp')

        # node
        for i, space_name_i in enumerate(space_node):
            with g.subgraph(name="cluster" + space_name_i) as c:
                c.attr(color='blue')
                for node_name in space_node[space_name_i]:
                    c.node(node_name + self.gene_space_separator + space_name_i, label=node_name)
                c.attr(label=space_name_i)

                c.attr('node', shape='diamond', style='filled', color='lightgrey')
                for edge_name in sim["M_"]:
                    # chemical edge
                    if self.chemical_flow_character in edge_name:
                        react_rule_name, space_name = edge_name.split(self.gene_space_separator)
                        if space_name_i == space_name:
                            c.node(edge_name, label=react_rule_name)

        # edge
        for edge_name in sim["M_"]:
            # chemical edge
            if self.chemical_flow_character in edge_name:
                react_rule_name, space_name = edge_name.split(self.gene_space_separator)
                reactant_name_0 = to_string(
                    react_rules[react_rule_name]['rule'][0]) + self.gene_space_separator + space_name
                reactant_name_1 = to_string(
                    react_rules[react_rule_name]['rule'][1]) + self.gene_space_separator + space_name
                component_name = to_string(
                    react_rules[react_rule_name]['rule'][2]) + self.gene_space_separator + space_name

                g.edge(reactant_name_0, edge_name)
                g.edge(reactant_name_1, edge_name)
                g.edge(component_name, edge_name)

            else:
                diff_or_hamilt, node_name, space_name_0, space_name_1 = edge_name.split(self.gene_space_separator)
                node_name_0 = node_name + self.gene_space_separator + space_name_0
                node_name_1 = node_name + self.gene_space_separator + space_name_1
                if diff_or_hamilt == self.diffusion_flow_character:
                    g.edge(node_name_0, node_name_1, style="dashed")
                if diff_or_hamilt == self.hamiltonian_flow_character:
                    g.edge(node_name_0, node_name_1, style="bold")

        display(g)
        # g.view()

        ## simulate condition
        if len(sim["M_"]) == 0: sys.exit()

        ## pandas -> numpy
        for v in sim:
            sim[v] = sim[v].to_numpy()

        sim["n"] = sim["x_0"].shape[0]
        sim["c"] = sim["m_c"].shape[0]
        sim["e"] = sim["M_"].shape[1]

        system = System(sim)
        xp0 = np.array([sim["x_0"], sim["p_0"]]).reshape(-1)
        t = np.linspace(0, 10)
        xp = odeint(system.ode, xp0, t)


        def get_cmap(n, name='hsv'):
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(len(node))

        for i,node_name in enumerate(node) :
            plt.plot(t, xp[:, i], color=cmap(i), label=node_name)
        plt.xlabel('time')
        plt.legend(loc='best')
        plt.show()


        print(xp[0,:])

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



