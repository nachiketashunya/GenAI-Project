import subprocess
import os
import yaml

data_root = "/scratch/data/m23csa016"
output_parent = os.path.join(data_root, "meesho_data/finetuning")

# Base directory for YAML file and results
code_root = "/iitjhome/m23csa016"
base_dir = os.path.join(code_root, "meesho_code/cat_specific_attrs_fine")
yaml_path = os.path.join(base_dir, "cat_specific_fine.yml")

# List of datasets and their corresponding output directories
datasets = [
    ("cs_kurtis", os.path.join(output_parent, "cs_kurtis")),
    # ("cs_men_tshirts", os.path.join(output_parent, "cs_men_tshirts_cp")),
    ("cs_sarees", os.path.join(output_parent, "cs_sarees")),
    ("cs_women_tops", os.path.join(output_parent, "cs_women_tops")),
    ("cs_women_tshirts", os.path.join(output_parent, "cs_women_tshirts"))
]

# CUDA devices to use
cuda_devices = "0"

def run_training(config, dataset, output_dir):
    # Create a temporary YAML file with modified configuration
    temp_yaml_path = os.path.join(base_dir, f"temp_{dataset}.yml")
    config['dataset'] = dataset
    config['output_dir'] = output_dir
    
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run the training command
    command = f"llamafactory-cli train {temp_yaml_path}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")
    
    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"Started training for {dataset}")
        process.wait()
        print(f"Finished training for {dataset}")
    
    # Remove the temporary YAML file
    os.remove(temp_yaml_path)

# Load the base configuration
with open(yaml_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Run training for each dataset
for dataset, output_dir in datasets:
    run_training(base_config.copy(), dataset, output_dir)

print("All training jobs completed.")