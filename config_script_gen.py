import sys
import json
import os

required_folders = ["./config_folder", "./scripts", "./logs"]

for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

clustering_approaches = ["gm"]
epochs = [1, 5, 10]
loss_functions = ["cos"]
umap_usages = [False]
oracles = ["v1", "v1v2"]
datasets = {'20newsgroups':20,
'wvsh':2,
'nova':2,
'autovsaviation':2,
'simvsreal':2
            }
# Data to be written
for clustering_approach in clustering_approaches:
    for epoch in epochs:
        for loss_function in loss_functions:
            for umap_usage in umap_usages:
                for oracle in oracles:
                    for dataset in datasets:
                        # Generate config 
                        suffix = f"{dataset}{oracle}{clustering_approach}_{epoch}_{loss_function}_{umap_usage}"
                        dictionary = {
                                        "num_iters": 50,
                                        "model_path": "sentence-transformers/all-MiniLM-L6-v2",
                                        #"model_path": "saved/minilm-download",#saved/all-distilroberta-v1",
                                        "min_cluster_size": datasets[dataset],
                                        "clus_method":clustering_approach,
                                        "epoch": epoch,
                                        "loss": loss_function,
                                        "umap": umap_usage,
                                        "workdir":f'workdirs/run_{suffix}',
                                        "oracle": oracle,
                                        "dataset":dataset
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
                
