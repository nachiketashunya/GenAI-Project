import csv
import json
import os

# Paths
csv_file_path = '/iitjhome/m23csa016/meesho_code/all_attrs_fine/all_attrs.csv'  # Path to your CSV file
image_folder_path = '/scratch/data/m23csa016/meesho_data/train_images'  # Folder where your images are stored
output_json_file = '/iitjhome/m23csa016/meesho_code/all_attrs_fine/all_attrs.json'  # Output JSON file

# Initialize list to store the dataset
dataset = []

category_class_attribute_mapping = {
    'Kurtis': {
        'color': 'attr_1',
        'fit_shape': 'attr_2',
        'length': 'attr_3',
        'occasion': 'attr_4',
        'ornamentation': 'attr_5',
        'pattern': 'attr_6',
        'print_or_pattern_type': 'attr_7',
        'sleeve_length': 'attr_8',
        'sleeve_styling': 'attr_9'
    },
    'Men Tshirts': {
        'color': 'attr_1',
        'neck': 'attr_2',
        'pattern': 'attr_3',
        'print_or_pattern_type': 'attr_4',
        'sleeve_length': 'attr_5'
    },
    'Sarees': {
        'blouse_pattern': 'attr_1',
        'border': 'attr_2',
        'border_width': 'attr_3',
        'color': 'attr_4',
        'occasion': 'attr_5',
        'ornamentation': 'attr_6',
        'pallu_details': 'attr_7',
        'pattern': 'attr_8',
        'print_or_pattern_type': 'attr_9',
        'transparency': 'attr_10'
    },
    'Women Tops & Tunics': {
        'color': 'attr_1',
        'fit_shape': 'attr_2',
        'length': 'attr_3',
        'neck_collar': 'attr_4',
        'occasion': 'attr_5',
        'pattern': 'attr_6',
        'print_or_pattern_type': 'attr_7',
        'sleeve_length': 'attr_8',
        'sleeve_styling': 'attr_9',
        'surface_styling': 'attr_10'
    },
    'Women Tshirts': {
        'color': 'attr_1',
        'fit_shape': 'attr_2',
        'length': 'attr_3',
        'pattern': 'attr_4',
        'print_or_pattern_type': 'attr_5',
        'sleeve_length': 'attr_6',
        'sleeve_styling': 'attr_7',
        'surface_styling': 'attr_8'
    }
}

dataset = []

with open(csv_file_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        # Extract image filename from the image link
        
        image_filename = str(row['id']).zfill(6) + '.jpg'
        
        # Construct the image path in your train folder
        image_path = os.path.join(image_folder_path, image_filename)

        if os.path.exists(image_path):
            category = row['Category']
            
            if category in category_class_attribute_mapping:
                attributes = category_class_attribute_mapping[category]
                
                # Generate prompt
                prompt_attributes = ", ".join(attributes.keys())
                prompt = f"<image>Given this product image of '{category}' category, what are the {prompt_attributes} of the product?"
                
                # Generate content
                content_attrs = []
                for attr, attr_key in attributes.items():
                    attr_value = row[attr_key].strip()  # Remove any leading/trailing whitespace
                    content_attrs.append(f"'{attr_value}'")  # Wrap each value in quotes

                content = ", ".join(content_attrs) 
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