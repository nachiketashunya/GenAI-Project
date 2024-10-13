import subprocess
import os

# List of YAML files and their corresponding output directories
code_dir = "/iitjhome/m23csa016/meesho_code"
data_dir = "/scratch/data/m23csa016/meesho_data/finetuning"

# List of YAML files and their corresponding output directories
yaml_files = [
    (os.path.join(code_dir, "universal_attrs_fine/universal_fine.yml"), os.path.join(data_dir, "universal_attrs")),
    (os.path.join(code_dir, "women_group_attrs_fine/women_group_fine.yml"), os.path.join(data_dir, "women_group_attrs"))
]

def run_training(yaml_file, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "results.txt")
    
    # Run the training command and save output to results.txt
    command = f"llamafactory-cli train {yaml_file}"
    print(f"Running: {command}")
    
    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        process.wait()  # Ensure the next command runs only after the current one finishes
    
    print(f"Finished training for {yaml_file}, results saved in {output_file}")

# Execute training for each YAML file and save results to respective directories
for yaml_file, output_dir in yaml_files:
    run_training(yaml_file, output_dir)

print("All training jobs completed.")
