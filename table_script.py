import os
import json
from prettytable import PrettyTable
import glob
import pickle as pkl

x = PrettyTable()
x.field_names = ["Dataset", "Oracle", "Loss", "Clus Method", "Epoch per feedback", "Umap flag", *map(lambda i:f'F{i}',range(50))]
json_data = []
for filename in glob.glob('/home/bh563857/thesis_backend/config_folder/*.json'):
    with open(filename) as json_file:
        json_data = json.load(json_file)  
        r_file = f'{json_data["workdir"]}/result.pkl'.replace('workdirs','workdirs')
        if not(os.path.exists(r_file)):
            continue
        object = pkl.load(open(r_file,'rb')) 
        r = []
        for p in object['purities']:
            avg = p.mean(axis=0)[0]
            res_std = p.std(axis=0)[0]
            r.append(f'{avg:0.2f}±{res_std:0.2f}')
        r.extend(['NA']*(50-len(r)))
        x.add_row([json_data['dataset'],json_data['oracle'],json_data['loss'],json_data['clus_method'],json_data['epoch'], json_data['umap'], *r])
print(x.get_string(sortby="Clus Method"))



