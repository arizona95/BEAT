import numpy as np
import pandas as pd
import itertools
import graphviz
from IPython.display import display

class Gene :
    def __init__(self, net, param):

        self.net = net
        self.g_c = param["g_c"]
        self.g_g = param["g_c"]+1
        self.g_s = param["g_s"]
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

        self.output_controll_constant = {
            # name: [a, b] -> (a-b)*output + b : a ~ b
            'm_c' : [1, 0.01],                          # mass of component
            'q_c' : [0.01, -0.01],                      # charge of component
            'r_s' : [self.max_state, -self.max_state],  # react and state decision
            'k'   : [1, 0],                             # react rate
            'A'   : [4,0],                              # enthalpy of gene
            'S'   : [1,-1],                             # space exist decision
            'V'   : [1, 0],                             # volume matrix
            'D'   : [2, 0],                             # distance matrix
            'x0'  : [10, -10],                          # initial value
            'c_'  : [100, 10],                          # collision value
            'x_h' : [10, -0.01],                        # externel homeostasis value
            'h'   : [1, 0],                             # homeostasis speed value
            'diff' : [1000, -1000],                     # diffusion speed value
            'hamilt': [10, -10],                        # hamilt speed value

        }


    def expression(self):

        def to_string(list) :
            return self.vector_separator.join(str(e) for e in list)

        def net_output(input_vector):
            return self.net.activate([np.array(input_vector)]).numpy()[0]

        def rescale(value, type) :
            a = self.output_controll_constant[type][0]
            b = self.output_controll_constant[type][1]
            return (a-b)*value +b

        # make zero vector
        c_0 = [0] * self.g_c
        g_0 = [0] * self.g_g
        s_0 = [0] * self.g_s

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
        for i in range(self.g_c):
            c_vector = [0] * self.g_c
            c_vector[i] = 1
            input_vector = c_vector + g_0 + s_0 + g_0 + s_0
            output = net_output(input_vector)
            m_c.append(rescale(output[0], "m_c"))
            q_c.append(rescale(output[1], "q_c"))

        # 1-2. make init of Gene Map G0
        react_rules = dict()
        G0 = list()
        for i in range(self.g_c):
            add_G0 = [0] * self.g_g
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
                    react_or_not = rescale(output[2], "r_s")
                    react_rate = rescale(output[3], "k")
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
            A[to_string(gi)] = rescale(output[4], "A")

        # 2. Generation Space Map S_SET
        S = list()
        S_substrate = list()
        for neuron_idx in range(self.neuron_num):
            list(map(lambda x: S.append([neuron_idx] + list(x)), list(itertools.product([0, 1], repeat=self.g_s-1))))

        for si in S_substrate:
            input_vector = c_0 + g_0 + si + g_0 + s_0
            output = net_output(input_vector)
            space_or_not = rescale(output[5], "S")
            if space_or_not >= 0:
                S.append(si)

        ## make V matrix

        V = dict()
        self.V = V
        for i, si in enumerate(S):
            ## fs(si) = f(0,si,0,0)[3]
            input_vector = c_0 + g_0 + si + g_0 + s_0
            output = net_output(input_vector)
            V[to_string(si)] = rescale(output[6], "V")


        ## make D matrix

        D = [[0 for x in range(len(S))] for y in range(len(S))]
        for i, si in enumerate(S):
            for j, sj in enumerate(S):
                if i != j:
                    input_vector = c_0 + g_0 + si + g_0 + sj
                    output = net_output(input_vector)
                    distance = rescale(output[7], "D")
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
                x0 = rescale(output[8], "x0")
                c_ = rescale(output[9], "c_")
                xh = rescale(output[10], "x_h")
                h = rescale(output[11], "h")
                a = -np.log( A[to_string(gi)] * V[to_string(sj)] + 0.01)-10
                if xh <= 0:
                    h = 0
                    xh = 0
                if x0 > 0:
                    node_name = to_string(gi) + gene_space_separator + to_string(sj)
                    node[node_name] = [
                        gi,  # g
                        sj,  # s
                        x0,  # x0
                        gi[:-1],  # c
                        c_,  # c0
                        a,  # a
                        xh,  # xh
                        h,  # h
                    ]

        ## Generation Edge

        M_T = dict()
        k = dict()
        v = dict()
        er, ed, eh = 0, 0, 0

        ## chemical flow : same space
        space_node = dict()
        for i, si in enumerate(S):
            space_node[to_string(si)] = dict()

        for node_name in node:
            space_node[to_string(node[node_name][1])][to_string(node[node_name][0])] = node[node_name][0]

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

                    # count plus
                    er += 1

        ## diffusion flow : different space
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
                                diff_k = rescale(output[12], "diff")
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

                                    # count plus
                                    ed += 1

        ## hamiltonian flow : different space
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
                                # hamilt edge
                                hamilt_k = rescale(output[13], "hamilt")
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

                                    # count plus
                                    eh += 1

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
            self.model["x_0"][node_name] = node[node_name][2]
            self.model["p_0"][node_name] = 0
            self.model["M"][node_name] = node[node_name][3]

            add_S = dict()
            for i, si in enumerate(S):
                if to_string(si) == to_string(node[node_name][1]):
                    add_S[to_string(si)] = 1
                else:
                    add_S[to_string(si)] = 0
            self.model["S"][node_name] = add_S
            self.model["a"][node_name] = node[node_name][5]
            self.model["c_"][node_name] = node[node_name][4]
            self.model["x_h"][node_name] = node[node_name][6]
            self.model["h"][node_name] = node[node_name][7]

        self.model["x_0"] = pd.DataFrame.from_dict([self.model["x_0"]]).T
        self.model["p_0"] = pd.DataFrame.from_dict([self.model["p_0"]]).T
        self.model["M"] = pd.DataFrame.from_dict(self.model["M"]).T
        try: self.model["M_"] = pd.DataFrame(M_T).T[node.keys()].T
        except : self.model["M_"] = pd.DataFrame(M_T)
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
        self.model["c"] = self.g_c
        self.model["e"] = self.model["M_"].shape[1]
        self.model["er"] = er
        self.model["ed"] = ed
        self.model["eh"] = eh
        self.model["s"] = len(S)
        self.model["g_c"] = self.g_c
        self.model["g_s"] = self.g_s
        self.model["neuron_num"] = self.neuron_num
        self.model["input_num"] = self.input_num
        self.model["output_num"] = self.output_num
        self.model["node_names"] = list(self.model["x_0"].T.keys())


        input_maker = np.zeros((self.model["n"], self.neuron_num))

        self.extenel_signal_node = [0]* self.g_g
        self.extenel_signal_node[0] = 1

        self.extenel_signal_splace_check = [0]* (self.g_s - 1)

        for i, node_name in enumerate(self.model["x_0"].T):
            gene, space = node_name.split(self.gene_space_separator)
            gene_vector = list(map(lambda x:int(x), gene.split(self.vector_separator)))
            space_vector = list(map(lambda x:int(x),space.split(self.vector_separator)))
            if gene_vector == self.extenel_signal_node and space_vector[1:] == self.extenel_signal_splace_check:
                input_maker[i][space_vector[0]] = 1

        self.model["input_maker"] = input_maker

    def model_display(self, savePath=False):

        self.edge_color = "blue"
        self.input_color = "darkgreen"
        self.output_color = "darkorange"

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
        for i, space_name in enumerate(space_node):
            with g.subgraph(name="cluster" + space_name) as c:
                c.attr(color=self.edge_color)
                for node_name in space_node[space_name]:
                    c.attr('node', shape='circle', color='black')
                    gene_vector = list(map(lambda x: int(x), node_name.split(self.vector_separator)))
                    space_vector = list(map(lambda x: int(x), space_name.split(self.vector_separator)))
                    if gene_vector == self.extenel_signal_node and space_vector[1:] == self.extenel_signal_splace_check:
                        #input node
                        if 0 <= space_vector[0] and  space_vector[0] < self.input_num:
                            c.attr('node', shape='doublecircle', color=self.input_color)

                        #output node
                        elif  self.neuron_num - self.output_num <= space_vector[0] and  space_vector[0] < self.neuron_num:
                            c.attr('node', shape='doublecircle', color=self.output_color)

                    c.node(node_name + gene_space_separator + space_name, label=node_name)
                c.attr(label=space_name)

                c.attr('node', shape='square', style='filled', color='lightgrey', width="0.47")
                for edge_name in self.model["M_"]:
                    # chemical edge
                    if chemical_flow_character in edge_name:
                        react_rule_name, react_space_name = edge_name.split(gene_space_separator)
                        if space_name == react_space_name:
                            c.node(edge_name, label=react_rule_name)

        # space edge

        g.attr(compound='true')
        for i, space_name_i in enumerate(space_node):
            for j, space_name_j in enumerate(space_node):
                if i>j and self.model["D"][space_name_i][space_name_j]<=1 :
                    g.edge("cluster" + space_name_i, "cluster" + space_name_j, color=self.edge_color)

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
                    g.edge(node_name_0, node_name_1, style="bold")
                if diff_or_hamilt == hamiltonian_flow_character:
                    g.edge(node_name_0, node_name_1, style="dashed")

        g.render(filename=f"{savePath}\\graph", view=False, format ='png') #format : pdf, png

        #g.write_png(f"{savePath}\\graph.png")
        if not savePath : display(g)
        #g.view()