import os 
import glob 
for i in list(glob.glob("scripts/*autovsaviation*")):#+list(glob.glob("scripts/*_km_*_cos_False.*")):
    os.system(f'sbatch {i}')