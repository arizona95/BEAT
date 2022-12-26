import os
import sys
import json
from simulator import Simulator
import pandas as pd
import glob
from dominate import document
from dominate.tags import *
import webbrowser
import numpy as np
from datetime import datetime
import traceback

def log(text):
    with open('log.txt', 'a') as f:
        f.write(text + '\n')


class web_parser :
    def __init__(self):
        self.reaction_node_type = "Reaction"
        self.space_node_type = "Space"
        self.element_node_type = "Element"
        self.external_node_type = "External"

        self.reaction_edge_type = "ChemicalReaction"
        self.space_element_edge_type = "SpaceElement"
        self.space_neighbor_edge_type = "SpaceNeighbor"
        self.consist_edge_type = "Consist"
        self.hamiltonian_diffusion_edge_type = "HamiltonianDiffusion"
        self.external_diffusion_edge_type = "ExternalDiffusion"

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

        self.input_list_filename = list()

    def parse(self, data):

        self.debug = dict()
        json_data = json.loads(data)
        nodes_data = sorted(json_data["nodes"], key=lambda item: item['text'])


        sub_node_type = [self.reaction_node_type, self.space_node_type, self.element_node_type, self.external_node_type]


        id_to_node = dict()
        for node_data in nodes_data :
            id_to_node[node_data['id']] = node_data

        self.debug["id_to_node"] = id_to_node
        #node separate
        space_node = dict()
        reaction_node = dict()
        element_node = dict()
        external_node = dict()
        real_node = dict()
        for node_data in nodes_data :
            if node_data['type'] == self.space_node_type : space_node[node_data['text']] = node_data
            elif node_data['type'] == self.reaction_node_type : reaction_node[node_data['text']] = node_data
            elif node_data['type'] == self.element_node_type : element_node[node_data['text']] = node_data
            elif node_data['type'] == self.external_node_type : external_node[node_data['text']] = node_data
            else : real_node[node_data['text']] = node_data

        self.model['er'] = 0
        self.model['ed'] = 0
        self.model['eh'] = 0

        volume = dict()
        for node_data in nodes_data:
            if node_data['type'] not in sub_node_type :
                for out_node in node_data['pins']['out']['links']:
                    if 'pins' in node_data :
                        edge_type = out_node['type']
                        if edge_type == self.space_element_edge_type:
                            volume[node_data['text']] = float(id_to_node[out_node['node']]['data']['V'])

        #initialize 2D matrix
        for node_name in real_node:
            self.model['M_'][node_name] = dict()
            self.model['M'][node_name] = dict()
            self.model['S'][node_name] = dict()

        for node_name in space_node :
            self.model['D'][node_name] = dict()

        for node_name in real_node:
            self.model['p_0'][node_name]= 0
            self.model['x_h'][node_name] = 0
            self.model['h'][node_name] = 0

            if "input" in real_node[node_name]['data'] :
                if real_node[node_name]['data']["input"] != "" :
                    self.input_list_filename.append([node_name, real_node[node_name]['data']["input"]])

            try :
                self.model['x_0'][node_name]= float(real_node[node_name]['data']['C_0']) * volume[node_name]
            except :
                print(f"node name \"{node_name}\" has problem with \"C_0\" (initialize data)")
                exit(0)

            try:
                self.model['a'][node_name]= float(real_node[node_name]['data']['A'])/10-1-np.log(volume[node_name])
            except :
                print(f"node name \"{node_name}\" has problem with \"A\" (energy constant)")
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
                self.model['er']+=1
                for node_name in real_node :
                    self.model['M_'][node_name][node_data['text']] = 0

                for out_node in node_data['pins']['out']['links'] :
                    if id_to_node[out_node['node']]['text'] in self.model['M_'] :
                        self.model['M_'][id_to_node[out_node['node']]['text']][node_data['text']] += float(out_node['data']['amount'])
                        self.model['k'][node_data['text']] = float(node_data['data']['k'])
                        self.model['v'][node_data['text']] = 0

                for out_node in node_data['pins']['out']['links'] :
                    if id_to_node[out_node['node']]['text'] not in self.model['M_'] :
                        self.model['k'][node_data['text']] *= float(id_to_node[out_node['node']]['data']['V'])

            elif node_data['type'] == self.space_node_type :
                for node_name in real_node :
                    self.model['S'][node_name][node_data['text']] = 0

                for node_name in space_node :
                    self.model['D'][node_name][node_data['text']] = 0

            elif node_data['type'] == self.element_node_type:
                self.model['m_c'][node_data['text']] = float(node_data['data']['m'])
                self.model['q_c'][node_data['text']] = float(node_data['data']['q'])/10


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
                                self.model['eh'] += 1
                                #hamiltonian
                                edge_name = f"h_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                for node_name in real_node:
                                    self.model['M_'][node_name][edge_name] = 0

                                self.model['M_'][node_data['text']][edge_name] -= 1
                                self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1
                                self.model['k'][edge_name]=0
                                self.model['v'][edge_name] = float(out_node['data']['v']) * np.sqrt(volume[node_data['text']] * volume[id_to_node[out_node['node']]['text']])

                            if float(out_node['data']['k']) != 0 :
                                self.model['ed'] += 1
                                #diffusion
                                edge_name = f"d_{node_data['text']}_{id_to_node[out_node['node']]['text']}"

                                for node_name in real_node:
                                    self.model['M_'][node_name][edge_name] = 0

                                self.model['M_'][node_data['text']][edge_name] -= 1
                                self.model['M_'][id_to_node[out_node['node']]['text']][edge_name] += 1
                                self.model['k'][edge_name] = float(out_node['data']['k']) * np.sqrt(volume[node_data['text']] * volume[id_to_node[out_node['node']]['text']])
                                self.model['v'][edge_name] = 0


                        elif edge_type == self.space_element_edge_type :
                            self.model['S'][node_data['text']][id_to_node[out_node['node']]['text']] = 1

                        elif edge_type == self.external_diffusion_edge_type :
                            self.model['h'][node_data['text']] = float(out_node['data']['h'])
                            #assert one node linked
                            self.model['x_h'][node_data['text']] = float(id_to_node[out_node['node']]['data']['x_h']) * volume[node_data['text']]



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

        for r in self.model:
            if r not in ['node_names']:
                print(f"\n@@@@ {r} @@@@")
                print(self.model[r])

        return self.model



if __name__ == "__main__" :

    with open('log.txt', 'w') as f:
        f.write(f"simlate result\n")

    now_time = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
    os.makedirs(f"graph/result/{now_time}/")

    try :
        system_argv = {
            0: "phy",
            1: "bio"
        }

        ode_langage_argv = {
            0: "FORTRAN",
            1: "PYTHON_scipy_LSODA",
            2: "PYTHON_scipy_BDF",
            3: "PYTHON_scipy_Radau",
            4: "PYTHON_scipy_DOP853",
            5: "PYTHON_scipy_RK23",
            6: "PYTHON_scipy_RK45",
            7: "PYTHON_scikits_bdf",
            8: "PYTHON_scikits_rk5",
            9: "PYTHON_scikits_rk8",
            10: "PYTHON_scikits_beuler",
            11: "PYTHON_scikits_trapz",
            12: "JULIA",
            13: "JULIA_NUMBA"
        }

        with open("graph/Editor/args.txt", 'r') as args_file :
            args_txt = args_file.readlines()

        state = json.loads(args_txt[2].replace("$","\""))
        running_time = float(state["time"])
        parser = web_parser()
        model = parser.parse(args_txt[0])
        engine = system_argv[int(state['system'])]
        log(f"engine: {engine}")
        ode_language = ode_langage_argv[int(state['language'])]
        log(f"ode_language: {ode_language}")
        simulator = Simulator(model, engine=engine, ode_language=ode_language)
        selected_str = args_txt[1].replace('$','').replace('\n','')
        log(f"selected: {selected_str}")
        if len(selected_str) == 0 : selected = None
        else :selected = [parser.debug["id_to_node"][int(i)]['text'] for i in selected_str.split(',')[:-1]]
        input_list = list()
        for node_name, filename in parser.input_list_filename :
            df = pd.read_excel(str(filename), sheet_name="input", engine='openpyxl' ,usecols=['time','C_t'])
            for i in range(len(df)) :
                append_input = [node_name, df['time'][i], df['C_t'][i]]
                input_list.append(append_input)

        input_list.sort(key=lambda x: x[1])
        r_t=0
        for node_name, input_time, c_t in input_list :
            if input_time - r_t !=0  and  running_time >= input_time :
                success = simulator.run(input_time - r_t, selectNode=selected)
                r_t = input_time
                simulator.input_by_node_name(node_name, c_t)

        success = simulator.run(running_time-r_t, selectNode=selected, labelRecord=True)
        simulator.visualize(savePath=f"graph/result/{now_time}/")

    except Exception as e:
        log(str(traceback.format_exc()))


    #log(str(simulator.history["xp"][-1,:]))

    with open('log.txt', 'r') as f:
        log_texts =f.readlines()

    try:
        with open('time_log.txt', 'r') as f:
            log_texts.append(f"{f.read()} seconds taken for simulating")
        os.remove('time_log.txt')
    except : pass

    with document(title='Photos') as doc:
        for log_text in log_texts :
            h3(log_text)
        for path in glob.glob(f"graph/result/{now_time}/model*.png"):
            div(img(src=path.split('\\')[-1], width="100%"), _class='photo')

    with open(f'graph/result/{now_time}/gallery.html', 'w') as f:
        f.write(doc.render())


    webbrowser.open('file://' + os.path.realpath(f"graph/result/{now_time}/gallery.html"), new=2)


## check reaction consister
## simulation ~~