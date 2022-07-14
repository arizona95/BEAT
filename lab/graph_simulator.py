import os
import json

with open('graph/data/data.json') as json_file :
    json_data = json.load(json_file)

    for key in json_data :
        print(key)



## check reaction consister
## simulation ~~