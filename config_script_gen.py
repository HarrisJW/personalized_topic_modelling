import sys
import json
import os

required_folders = ["./config_folder", "./scripts", "./logs"]

for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

c = ["gm"] 
e = [1,5,10]
l = ["cos"]
u = [False]
o = ["v1","v1v2"]
d = {'20newsgroups':20,
'wvsh':2,
'nova':2,
'autovsaviation':2,
'simvsreal':2
}
# Data to be written

for i in c:
    for j in e:
        for k in l:
            for m in u:
                for n in o:
                    for a in d:
                        # Generate config 
                        suffix = f"{a}{n}{i}_{j}_{k}_{m}"
                        dictionary = {
                                        "num_iters": 50,
                                        "model_path": "saved/minilm-download",#saved/all-distilroberta-v1",
                                        "min_cluster_size": d[a],
                                        "clus_method":i,
                                        "epoch": j,
                                        "loss": k,
                                        "umap": m,
                                        "workdir":f'workdirs/run_{suffix}',
                                        "oracle": n,
                                        "dataset":a
                                    }
    
                        # Serializing json
                        json_object = json.dumps(dictionary, indent=4)
            
                        # Writing to sample.json
                        with open(f"config_folder/config_{suffix}.json", "w") as outfile:
                            outfile.write(json_object)
                        
                        # Generate script
                        with open(f"scripts/eval_{suffix}.sh",'w') as f:
                            f.write(f"""#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-emilios
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --out=logs/eval_{suffix}.log

source ./global_env/bin/activate
python -u quant_eval.py config_folder/config_{suffix}.json
""")
                
