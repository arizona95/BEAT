import os
import json
import pandas as pd

class web_parser :
    def __init__(self):
        self.reaction_node_type = "Reaction"

        self.reaction_edge_type = "ChemicalReaction"
        self.hamiltonian_edge_type = "Hamiltonian"
        self.diffusion_edge_type = "Diffusion"

        self.model = dict()
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

    def parse(self, filepath):
        with open(filepath) as json_file :
            json_data = json.load(json_file)


            for key in json_data :
                print(key)

            node_idx = dict()
            nodes_data = json_data["nodes"]

            id_to_node = dict()
            for node_data in nodes_data :
                id_to_node[node_data['id']] = node_data


            for node_data in nodes_data :
                #print(node_data)

                if node_data['type'] != self.reaction_node_type :
                    node_idx[node_data['text']] = len(node_idx)
                    self.model['M_'][node_data['text']] = dict()
                    try :
                        self.model['x_0'][node_data['text']]= node_data['data']['x_0']
                    except :
                        print(f"node name \"{node_data['text']}\" has problem with \"x_0\" (initialize data)")
                        exit(0)

                    try:
                        self.model['a'][node_data['text']]= node_data['data']['a']
                    except :
                        print(f"node name \"{node_data['text']}\" has problem with \"a\" (energy constant)")
                        exit(0)

                #break

            # make M_
            for node_data in nodes_data :
                #print(node_data)

                if node_data['type'] == self.reaction_node_type:
                    for node_name in node_idx :
                        self.model['M_'][node_name][node_data['text']] = 0

                    for out_node in node_data['pins']['out']['links'] :
                        self.model['M_'][id_to_node[out_node['node']]['text']][node_data['text']] += 1

            for node_data in nodes_data:
                if node_data['type'] != self.reaction_node_type:
                    if 'pins' in node_data :
                        for out_node in node_data['pins']['out']['links']:
                            if out_node['type'] == self.reaction_edge_type:
                                self.model['M_'][node_data['text']][id_to_node[out_node['node']]['text']] -=1

                            elif out_node['type'] == self.hamiltonian_edge_type :
                                edge_name = f"h_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                for node_name in node_idx:
                                    self.model['M_'][node_name][edge_name] = 0

                                self.model['M_'][node_data['text']][edge_name] -= 1
                                self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1

                            elif out_node['type'] == self.diffusion_edge_type:
                                edge_name = f"d_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                for node_name in node_idx:
                                    self.model['M_'][node_name][edge_name] = 0

                                self.model['M_'][node_data['text']][edge_name] -= 1
                                self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1

            pd_M_ = pd.DataFrame(self.model['M_']).T
            print(pd_M_)
            print(pd_M_.to_numpy())




        print(self.model['x_0'])

        print(self.model['a'])

        print(json_data['nodes'][0])
        print(json_data['nodes'][2])

        print(json_data['nodes'][5])
        print(node_idx)


if __name__ == "__main__" :
    parser = web_parser()
    parser.parse('graph/data/data.json')


## check reaction consister
## simulation ~~