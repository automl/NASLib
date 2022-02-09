# read yaml config as in nasbench for creating nice configs and easy access
import yaml
import os
from glob import glob

# Short workaround, such that I do not have to design a proper class for a small config file during development
# Allows easier access to config by attributes instead of dictionary key.
from addict import Dict

from create_configs import create_configs
# Read YAML file
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

def check_config(search_space, dataset, optimizer, trials, HPO, num_config):
    folder = os.path.join(".", "bbo", "configs_m1", search_space, dataset, optimizer)
    test = os.path.join(folder, "**", "*.yaml")  
    files = glob(test, recursive=True)
    num_config = num_config if HPO else 1
    if len(files) == trials:
        print("Created config file(s) successfully")
        return
    print("Config file(s) not successfully created")        

# convert such that config elements are accessiable via attributes
config = Dict(config)
start_seed = config.start_seed if config.start_seed else 0
trials = config.trials
end_seed = start_seed + trials - 1

optimizers = config.optimizers
# TODO: Implement check for optimizers
# See lines 13 - 18 in make_configs_nb201.sh

out_dir = config.out_dir

config_type = config.config_type
if config_type not in {'bbo-bs', 'predictor-bs'}:
    print('Invalid config')
    print('config_type either bbo-bs or predictor-bs')
    exit(1)

search_space = config.search_space

datasets = config.datasets
fidelity = config.fidelity
epochs = config.epochs

predictor = config.predictor

HPO = config.HPO
num_config = config.num_config

for dataset in datasets:
    for optimizer in optimizers:
        print(f"Creating config for dataset: {dataset}: & optimizer: {optimizer}")
        create_configs(
            start_seed=start_seed,
            trials=trials,
            out_dir=out_dir,
            dataset=dataset,
            config_type=config_type,
            search_space=search_space,
            optimizer=optimizer,
            predictor=predictor,
            fidelity=fidelity,
            epochs=epochs,
            HPO=HPO,
            num_config=num_config
        )
        check_config(search_space, dataset, optimizer, trials, HPO, num_config)

""" 
    --acq_fn_optimization $acq_fn_optimization --predictor $predictor
    TODO: acq_fn_optimization -- this needs to be done
"""