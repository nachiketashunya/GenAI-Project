import csv
import json
import os

# Paths
csv_file_path = '/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/cs_sarees_10k.csv'  # Path to your CSV file
image_folder_path = '/scratch/data/m23csa016/meesho_data/train_images'  # Folder where your images are stored
output_json_file = '/iitjhome/m23csa016/meesho_code/cat_specific_attrs_fine/cs_sarees_10k.json'  # Output JSON file

# Initialize list to store the dataset
dataset = []

# Open and read the CSV file
with open(csv_file_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        # Extract image filename from the image link
        
        image_filename = str(row['id']).zfill(6) + '.jpg'
        
        # Construct the image path in your train folder
        image_path = os.path.join(image_folder_path, image_filename)

        # Ensure the image exists in the train/ folder before adding to the dataset
        if os.path.exists(image_path):
            category = row['Category']
            blouse_pattern = row['blouse_pattern']
            border = row['border']
            border_width = row['border_width']
            occasion = row['occasion']
            ornamentation = row['ornamentation']
            pallu_details = row['pallu_details']
            transparency = row['transparency']

            prompt = f"<image>Analyze this {category} image and identify: blouse_pattern, border, border_width, occasion, ornamentation, pallu_details, transparency"

            content = f"{blouse_pattern}, {border}, {border_width}, {occasion}, {ornamentation}, {pallu_details}, {transparency}"

            # Construct the conversation in ShareGPT format
            conversation = {
                "messages": [
                    {
                        "content": prompt,
                        "role": "user"
                    },
                    {
                        "content": content,
                        "role": "assistant"
                    }
                ],
                "images": [
                    image_path
                ]
            }

            # Add the conversation to the dataset
            dataset.append(conversation)
        else:
            print(f"Image {image_filename} not found in {image_folder_path}, skipping.")

# Save the dataset to a JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

print(f"Dataset successfully saved to {output_json_file}")