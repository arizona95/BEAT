import numpy as np
import pandas as pd
import itertools
import graphviz
from IPython.display import display

class Gene :
    def __init__(self, net, param):

        self.net = net
        self.c_s = param["g_s"]-1
        self.g_s = param["g_s"]
        self.s_s = param["s_s"]
        self.max_state = param["max_state"]
        self.react_depth = param["react_depth"]
        self.neuron_num = param["neuron_num"]
        self.input_num = param["input_num"]
        self.output_num = param["output_num"]

        self.rule_num =0
        self.model = dict()
        self.set = dict()

        self.vector_separator = "/"
        self.gene_space_separator = "-"
        self.chemical_flow_character = "r"
        self.diffusion_flow_character = "d"
        self.hamiltonian_flow_character = "h"

    def expression(self):

        def to_string(list) :
            return self.vector_separator.join(str(e) for e in list)

        def net_output(input_vector):
            return self.net.activate([np.array(input_vector)]).numpy()[0]

        # make zero vector
        c_0 = [0] * self.c_s
        g_0 = [0] * self.g_s
        s_0 = [0] * self.s_s

        # make abbreviation
        vector_separator = self.vector_separator
        gene_space_separator = self.gene_space_separator
        chemical_flow_character = self.chemical_flow_character
        diffusion_flow_character = self.diffusion_flow_character
        hamiltonian_flow_character = self.hamiltonian_flow_character

        # 1. Generation Gene Map G_SET

        # 1-1. make mass and charge of component
        m_c = list()
        q_c = list()
        for i in range(self.c_s):
            c_vector = [0] * self.c_s
            c_vector[i] = 1
            input_vector = c_vector + g_0 + s_0 + g_0 + s_0
            output = net_output(input_vector)
            m_c.append(output[0])
            q_c.append((2 * output[1] - 1)/100)

        # 1-2. make init of Gene Map G0
        react_rules = dict()
        G0 = list()
        for i in range(self.c_s):
            add_G0 = [0] * self.g_s
            add_G0[i] = 1
            G0.append(add_G0)

        # 1-3. make G by reaction
        G = G0
        new_G = G0
        rule_num =0
        for depth in range(self.react_depth):
            add_G = list()
            for gi in G:
                for gj in new_G:
                    input_vector = c_0 + gi + s_0 + gj + s_0
                    output = net_output(input_vector)
                    react_or_not = self.max_state * (2*output[2]-1)
                    react_rate = output[3]
                    if react_or_not >= 0:
                        new_g = list(np.array(gi[:-1]) + np.array(gj[:-1])) + [int(react_or_not)]
                        add_G.append(new_g)
                        rule_num += 1
                        react_rules[self.chemical_flow_character + str(rule_num)] = {
                            "rule": [gi, gj, new_g],
                            "k": react_rate,
                        }
            G = G + add_G
            new_G = add_G

        # 1-4. make A dict
        A = dict()
        for i, gi in enumerate(G):
            input_vector = c_0 + gi + s_0 + g_0 + s_0
            output = net_output(input_vector)
            A[to_string(gi)] = 5 * output[4]

        # 2. Generation Space Map S_SET
        S = list()
        S_substrate = list()
        for neuron_idx in range(self.neuron_num):
            list(map(lambda x: S.append([neuron_idx] + list(x)), list(itertools.product([0, 1], repeat=self.s_s-1))))

        for si in S_substrate:
            input_vector = c_0 + g_0 + si + g_0 + s_0
            output = net_output(input_vector)
            space_or_not = 2 * output[5] - 1
            if space_or_not >= 0:
                S.append(si)

        ## make V matrix

        V = dict()
        for i, si in enumerate(S):
            ## fs(si) = f(0,si,0,0)[3]
            input_vector = c_0 + g_0 + si + g_0 + s_0
            output = net_output(input_vector)
            V[to_string(si)] = output[6]

        ## make D matrix

        D = [[0 for x in range(len(S))] for y in range(len(S))]
        for i, si in enumerate(S):
            for j, sj in enumerate(S):
                if i != j:
                    input_vector = c_0 + g_0 + si + g_0 + sj
                    output = net_output(input_vector)
                    distance = 2 * output[7]
                    if distance < 1:  ## neighbor
                        D[i][j] = distance
                        D[j][i] = distance
                    else:
                        D[i][j] = np.inf
                        D[j][i] = np.inf

        ## 3. Generation real node
        node = dict()  # name:[x0, c, s, a, xh, h]


        for i, gi in enumerate(G):
            for j, sj in enumerate(S):
                input_vector = c_0 + gi + sj + g_0 + s_0
                output = net_output(input_vector)
                x0 = 20 * output[8] - 10
                xh = 20 * output[9] - 10
                h = output[10]
                if xh <= 0:
                    h = 0
                    xh = 0
                if x0 > 0:
                    node_name = to_string(gi) + gene_space_separator + to_string(sj)
                    node[node_name] = [
                        x0,  # x0
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

        for node_name in node:
            space_node[to_string(node[node_name][2])][to_string(node[node_name][6])] = node[node_name][6]

        for react_rule_name in react_rules:
            react_rule = react_rules[react_rule_name]["rule"]
            for space_name in space_node:
                if to_string(react_rule[0]) in space_node[space_name] and \
                        to_string(react_rule[1]) in space_node[space_name] and \
                        to_string(react_rule[2]) in space_node[space_name]:

                    edge_name = react_rule_name + gene_space_separator + space_name

                    # add M_T
                    M_T_add = dict()
                    for node_name in node:
                        M_T_add[node_name] = 0
                    M_T_add[to_string(react_rule[0]) + gene_space_separator + space_name] += 1
                    M_T_add[to_string(react_rule[1]) + gene_space_separator + space_name] += 1
                    M_T_add[to_string(react_rule[2]) + gene_space_separator + space_name] = -1
                    M_T[edge_name] = M_T_add

                    # add k
                    k[edge_name] = react_rules[react_rule_name]["k"]

                    # add v
                    v[edge_name] = 0

        ## diffusion flow, hamiltonian flow : different space
        for i, space_name_i in enumerate(space_node):
            si = list(map(lambda x: int(x), space_name_i.split(vector_separator)))
            for j, space_name_j in enumerate(space_node):
                sj = list(map(lambda x: int(x), space_name_j.split(vector_separator)))
                # if neighbor space
                if i > j and D[i][j] > 0 and D[i][j] < 1:
                    for n, node_name_n in enumerate(space_node[space_name_i]):
                        for m, node_name_m in enumerate(space_node[space_name_j]):
                            if node_name_n == node_name_m:
                                input_vector = c_0 + space_node[space_name_i][node_name_n] + \
                                               si + space_node[space_name_j][node_name_m] + sj

                                ##print(f"input_vector: {input_vector}")
                                # diff edge
                                diff_k = 2 * float(net_output(input_vector)[11]) - 1
                                if diff_k > 0:  # edge connect

                                    edge_name = diffusion_flow_character + \
                                                gene_space_separator + node_name_n + \
                                                gene_space_separator + space_name_j + \
                                                gene_space_separator + space_name_i

                                    # add M_T
                                    M_T_add = dict()
                                    for node_name in node:
                                        M_T_add[node_name] = 0
                                    M_T_add[node_name_n + gene_space_separator + space_name_i] = 1
                                    M_T_add[node_name_m + gene_space_separator + space_name_j] = -1
                                    M_T[edge_name] = M_T_add

                                    # add k
                                    k[edge_name] = diff_k

                                    # add v
                                    v[edge_name] = 0

                                # hamilt edge
                                hamilt_k = 2 * float(net_output(input_vector)[12]) - 1
                                if hamilt_k > 0:  # edge connect

                                    edge_name = hamiltonian_flow_character + \
                                                gene_space_separator + node_name_n + \
                                                gene_space_separator + space_name_j + \
                                                gene_space_separator + space_name_i

                                    # add M_T
                                    M_T_add = dict()
                                    for node_name in node:
                                        M_T_add[node_name] = 0
                                    M_T_add[node_name_n + gene_space_separator + space_name_i] = 1
                                    M_T_add[node_name_m + gene_space_separator + space_name_j] = -1
                                    M_T[edge_name] = M_T_add

                                    # add k
                                    k[edge_name] = 0

                                    # add v
                                    v[edge_name] = hamilt_k

        self.model["x_0"] = dict()
        self.model["p_0"] = dict()
        self.model["M"] = dict()
        self.model["M_"] = dict()
        self.model["S"] = dict()
        self.model["D"] = dict()
        self.model["a"] = dict()
        self.model["k"] = dict()
        self.model["v"] = dict()
        self.model["c_"] = dict()
        self.model["x_h"] = dict()
        self.model["h"] = dict()

        for node_name in node:
            self.model["x_0"][node_name] = node[node_name][0]
            self.model["p_0"][node_name] = 0
            self.model["M"][node_name] = node[node_name][1]

            add_S = dict()
            for i, si in enumerate(S):
                if to_string(si) == to_string(node[node_name][2]):
                    add_S[to_string(si)] = 1
                else:
                    add_S[to_string(si)] = 0
            self.model["S"][node_name] = add_S
            self.model["a"][node_name] = node[node_name][3]
            self.model["c_"][node_name] = 1
            self.model["x_h"][node_name] = node[node_name][4]
            self.model["h"][node_name] = node[node_name][5]

        self.model["x_0"] = pd.DataFrame.from_dict([self.model["x_0"]]).T
        self.model["p_0"] = pd.DataFrame.from_dict([self.model["p_0"]]).T
        self.model["M"] = pd.DataFrame.from_dict(self.model["M"]).T
        self.model["M_"] = pd.DataFrame.from_dict(M_T)
        self.model["S"] = pd.DataFrame.from_dict(self.model["S"]).T
        space_names = list(map(lambda x: to_string(x), S))
        self.model["D"] = pd.DataFrame.from_dict(D)
        self.model["D"].columns = space_names
        self.model["D"] = self.model["D"].T
        self.model["D"].columns = space_names
        self.model["m_c"] = pd.DataFrame.from_dict([m_c]).T
        self.model["q_c"] = pd.DataFrame.from_dict([q_c]).T
        self.model["a"] = pd.DataFrame.from_dict([self.model["a"]]).T
        self.model["k"] = pd.DataFrame.from_dict([k]).T
        self.model["v"] = pd.DataFrame.from_dict([v]).T
        self.model["c_"] = pd.DataFrame.from_dict([self.model["c_"]]).T
        self.model["x_h"] = pd.DataFrame.from_dict([self.model["x_h"]]).T
        self.model["h"] = pd.DataFrame.from_dict([self.model["h"]]).T

        self.set["G"] = G
        self.set["A"] = A
        self.set["S"] = S
        self.set["node"] = node
        self.set["space_node"] = space_node
        self.set["react_rules"] = react_rules


        # model discription
        self.model["n"] = len(node)
        self.model["c_s"] = self.c_s
        self.model["s_s"] = self.s_s
        self.model["neuron_num"] = self.neuron_num
        self.model["input_num"] = self.input_num
        self.model["output_num"] = self.output_num
        self.model["node_names"] = list(self.model["x_0"].T.keys())


        input_maker = np.zeros((self.model["n"], self.neuron_num))

        extenel_signal_node = [0]* self.g_s
        extenel_signal_node[0] = 1

        extenel_signal_splace_check = [0]* (self.model["s_s"] - 1)

        for i, node_name in enumerate(self.model["x_0"].T):
            gene, space = node_name.split(self.gene_space_separator)
            gene_vector = list(map(lambda x:int(x), gene.split(self.vector_separator)))
            space_vector = list(map(lambda x:int(x),space.split(self.vector_separator)))
            if gene_vector == extenel_signal_node and space_vector[1:] == extenel_signal_splace_check:
                input_maker[i][space_vector[0]] = 1

        self.model["input_maker"] = input_maker



    def print_model_info(self):
        print(f"react_rules : {self.set['react_rules'] }")
        print(f"G set : {self.set['G']}")
        print(f"A set : {self.set['G']}")
        print(f"S set: {self.set['S']}")
        print(f"node list : \n")
        for node in self.set["node"] :
            print(f"{node} : {self.set['node'][node]}")

        for e in self.model:
            print(e)
            display(self.model[e])

    def model_display(self, savePath=False):

        def to_string(list) :
            return self.vector_separator.join(str(e) for e in list)

        # make abbreviation
        vector_separator = self.vector_separator
        gene_space_separator = self.gene_space_separator
        chemical_flow_character = self.chemical_flow_character
        diffusion_flow_character = self.diffusion_flow_character
        hamiltonian_flow_character = self.hamiltonian_flow_character

        ## graph display

        if not savePath : display(self.model["M_"])

        # model graph
        node = self.set["node"]
        space_node = self.set["space_node"]
        react_rules = self.set["react_rules"]

        g = graphviz.Graph('G', filename='../graph/test.gv', engine='fdp')
        for i, space_name_i in enumerate(space_node):
            with g.subgraph(name="cluster" + space_name_i) as c:
                c.attr(color='blue')
                for node_name in space_node[space_name_i]:
                    c.node(node_name + gene_space_separator + space_name_i, label=node_name)
                c.attr(label=space_name_i)

                c.attr('node', shape='diamond', style='filled', color='lightgrey')
                for edge_name in self.model["M_"]:
                    # chemical edge
                    if chemical_flow_character in edge_name:
                        react_rule_name, space_name = edge_name.split(gene_space_separator)
                        if space_name_i == space_name:
                            c.node(edge_name, label=react_rule_name)

        # edge
        for edge_name in self.model["M_"]:
            # chemical edge
            if chemical_flow_character in edge_name:
                react_rule_name, space_name = edge_name.split(gene_space_separator)
                reactant_name_0 = to_string(
                    react_rules[react_rule_name]['rule'][0]) + gene_space_separator + space_name
                reactant_name_1 = to_string(
                    react_rules[react_rule_name]['rule'][1]) + gene_space_separator + space_name
                component_name = to_string(
                    react_rules[react_rule_name]['rule'][2]) + gene_space_separator + space_name

                g.edge(reactant_name_0, edge_name)
                g.edge(reactant_name_1, edge_name)
                g.edge(component_name, edge_name)

            else:
                diff_or_hamilt, node_name, space_name_0, space_name_1 = edge_name.split(gene_space_separator)
                node_name_0 = node_name + gene_space_separator + space_name_0
                node_name_1 = node_name + gene_space_separator + space_name_1
                if diff_or_hamilt == diffusion_flow_character:
                    g.edge(node_name_0, node_name_1, style="dashed")
                if diff_or_hamilt == hamiltonian_flow_character:
                    g.edge(node_name_0, node_name_1, style="bold")

        g.render(filename=f"{savePath}\\graph.dot", view=False)

        #g.write_png(f"{savePath}\\graph.png")
        if not savePath : display(g)
        #g.view()