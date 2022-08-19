import os
import json
from simulator import Simulator
import pandas as pd
import webbrowser
class web_parser :
    def __init__(self):
        self.reaction_node_type = "Reaction"
        self.space_node_type = "Space"
        self.element_node_type = "Element"

        self.reaction_edge_type = "ChemicalReaction"
        self.hamiltonian_diffusion_edge_type = "HamiltonianDiffusion"
        self.space_element_edge_type = "SpaceElement"
        self.space_neighbor_edge_type = "SpaceNeighbor"
        self.consist_edge_type = "Consist"

        self.model = dict()
        self.model["x_0"] = dict()  #ok
        self.model["p_0"] = dict()  #ok
        self.model["M"] = dict()    #ok
        self.model["M_"] = dict()   #ok
        self.model["S"] = dict()    #ok
        self.model["D"] = dict()    #ok
        self.model["a"] = dict()    #ok
        self.model["k"] = dict()    #ok
        self.model["v"] = dict()    #ok
        self.model["c_"] = dict()   #ok
        self.model["x_h"] = dict()  #ok
        self.model["h"] = dict()    #ok
        self.model["m_c"] = dict()
        self.model["q_c"] = dict()

    def parse(self, filepath):

        with open(filepath) as json_file :
            json_data = json.load(json_file)
            nodes_data = sorted(json_data["nodes"], key=lambda item: item['text'])


            sub_node_type = [self.reaction_node_type, self.space_node_type, self.element_node_type]


            id_to_node = dict()
            for node_data in nodes_data :
                id_to_node[node_data['id']] = node_data

            #node seperate
            space_node = dict()
            reaction_node = dict()
            element_node = dict()
            real_node = dict()
            for node_data in nodes_data :
                if node_data['type'] == self.space_node_type : space_node[node_data['text']] = node_data
                elif node_data['type'] == self.reaction_node_type : reaction_node[node_data['text']] = node_data
                elif node_data['type'] == self.element_node_type : element_node[node_data['text']] = node_data
                else : real_node[node_data['text']] = node_data

            #initialize 2D matrix
            for node_name in real_node:
                self.model['M_'][node_name] = dict()
                self.model['M'][node_name] = dict()
                self.model['S'][node_name] = dict()

            for node_name in space_node :
                self.model['D'][node_name] = dict()

            #make x_0, a
            for node_name in real_node:
                self.model['p_0'][node_name]= 0
                self.model['x_h'][node_name] = 0
                self.model['h'][node_name] = 0

                try :
                    self.model['x_0'][node_name]= float(real_node[node_name]['data']['x_0'])
                except :
                    print(f"node name \"{node_name}\" has problem with \"x_0\" (initialize data)")
                    exit(0)

                try:
                    self.model['a'][node_name]= float(real_node[node_name]['data']['a'])
                except :
                    print(f"node name \"{node_name}\" has problem with \"a\" (energy constant)")
                    exit(0)

                try:
                    self.model['c_'][node_name]= float(real_node[node_name]['data']['c'])
                except :
                    print(f"node name \"{node_name}\" has problem with \"c\" (energy constant)")
                    exit(0)


            #prepare & setting
            for node_data in nodes_data :
                #print(node_data)

                if node_data['type'] == self.element_node_type :
                    for node_name in real_node :
                        self.model['M'][node_name][node_data['text']] = 0

                    for out_node in node_data['pins']['out']['links']:
                        self.model['M'][id_to_node[out_node['node']]['text']][node_data['text']] += float(out_node['data']['amount'])


                if node_data['type'] == self.reaction_node_type:
                    for node_name in real_node :
                        self.model['M_'][node_name][node_data['text']] = 0

                    for out_node in node_data['pins']['out']['links'] :
                        self.model['M_'][id_to_node[out_node['node']]['text']][node_data['text']] += float(out_node['data']['amount'])
                        self.model['k'][node_data['text']] = float(node_data['data']['k'])
                        self.model['v'][node_data['text']] = 0


                elif node_data['type'] == self.space_node_type :
                    for node_name in real_node :
                        self.model['S'][node_name][node_data['text']] = 0

                    for node_name in space_node :
                        self.model['D'][node_name][node_data['text']] = 0

                elif node_data['type'] == self.element_node_type:
                    self.model['m_c'][node_data['text']] = float(node_data['data']['m'])
                    self.model['q_c'][node_data['text']] = float(node_data['data']['q'])


            #setting
            for node_data in nodes_data:

                if node_data['type'] == self.space_node_type :
                    if 'pins' in node_data :
                        for out_node in node_data['pins']['out']['links']:
                            edge_type = out_node['type']
                            if edge_type == self.space_neighbor_edge_type:
                                self.model['D'][node_data['text']][id_to_node[out_node['node']]['text']] = float(out_node['data']['distance'])
                                self.model['D'][id_to_node[out_node['node']]['text']][node_data['text']] = float(out_node['data']['distance'])

                if node_data['type'] not in sub_node_type :
                    if 'pins' in node_data :
                        for out_node in node_data['pins']['out']['links']:
                            edge_type = out_node['type']
                            if edge_type == self.reaction_edge_type:
                                self.model['M_'][node_data['text']][id_to_node[out_node['node']]['text']] -= float(out_node['data']['amount'])

                            elif edge_type == self.hamiltonian_diffusion_edge_type :
                                if float(out_node['data']['v']) != 0 :
                                    #hamiltonian
                                    edge_name = f"h_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                    for node_name in real_node:
                                        self.model['M_'][node_name][edge_name] = 0

                                    self.model['M_'][node_data['text']][edge_name] -= 1
                                    self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1
                                    self.model['k'][edge_name]=0
                                    self.model['v'][edge_name] = float(out_node['data']['v'])

                                if float(out_node['data']['k']) != 0 :
                                    #diffusion
                                    edge_name = f"d_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                    for node_name in real_node:
                                        self.model['M_'][node_name][edge_name] = 0

                                    self.model['M_'][node_data['text']][edge_name] -= 1
                                    self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1
                                    self.model['k'][edge_name] = float(out_node['data']['k'])
                                    self.model['v'][edge_name] = 0


                            elif edge_type == self.space_element_edge_type :
                                self.model['S'][node_data['text']][id_to_node[out_node['node']]['text']] = 1


            #pd_M_ = pd.DataFrame(self.model['M_']).T
            #print("@M_")
            #print(pd_M_)
            #print(pd_M_.to_numpy())

            self.model["x_0"] = pd.DataFrame([self.model["x_0"]]).T
            self.model["p_0"] = pd.DataFrame([self.model["p_0"]]).T
            self.model["M"] =   pd.DataFrame(self.model["M"]).T
            self.model["M_"] =  pd.DataFrame(self.model["M_"]).T
            self.model["S"] =   pd.DataFrame(self.model["S"]).T
            self.model["D"] =   pd.DataFrame(self.model["D"]).T
            self.model["a"] =   pd.DataFrame([self.model["a"]]).T
            self.model["k"] =   pd.DataFrame([self.model["k"]]).T
            self.model["v"] =   pd.DataFrame([self.model["v"]]).T
            self.model["c_"] =  pd.DataFrame([self.model["c_"]]).T
            self.model["x_h"] = pd.DataFrame([self.model["x_h"]]).T
            self.model["h"] =   pd.DataFrame([self.model["h"]]).T
            self.model["m_c"] = pd.DataFrame([self.model["m_c"]]).T
            self.model["q_c"] = pd.DataFrame([self.model["q_c"]]).T


            self.model['n'] = len(real_node)
            self.model['c'] = len(element_node)
            self.model['e'] = len(self.model['k'])
            self.model['s'] = len(space_node)

            self.model['node_names'] = real_node

            for r in self.model :
                if r not in ['node_names']:
                    print(f"\n@@@@ {r} @@@@")
                    print(self.model[r])
            return self.model



if __name__ == "__main__" :
    parser = web_parser()
    graph_model_filename = "unit3"
    model = parser.parse(f"graph/data/{graph_model_filename}.json")
    simulator = Simulator(model, engine="phy", ode_language='PYTHON')
    success = simulator.run(2)
    simulator.visualize(savePath=f"graph/result/")
    webbrowser.open('file://' + os.path.realpath(f"graph/result/model_2.png"), new=2)


## check reaction consister
## simulation ~~