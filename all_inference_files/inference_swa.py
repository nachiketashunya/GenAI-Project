import torch
import pdb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import clip
from collections import defaultdict
import numpy as np
import os
from tqdm.auto import tqdm
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
import json
import pandas as pd
import os
from tqdm import tqdm
import torch
from PIL import Image
import clip
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip

import sys
# from clipvit_clf_unfreeze import CategoryAwareAttributePredictor

# Category to attribute mapping
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


class ImageDataset(Dataset):
    """Dataset class for batch processing of images"""
    def __init__(self, image_paths, categories, clip_preprocess):
        self.image_paths = image_paths
        self.categories = categories
        self.clip_preprocess = clip_preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image in case of error
            image = torch.zeros(3, 224, 224)
        return image, self.categories[idx]
  
class CategoryAwareAttributePredictor(nn.Module):
    def __init__(self, clip_dim=512, category_attributes=None, attribute_dims=None, hidden_dim=512, dropout_rate=0.2, num_hidden_layers=1):
        super(CategoryAwareAttributePredictor, self).__init__()
        
        self.category_attributes = category_attributes
        
        # Create prediction heads for each category-attribute combination
        self.attribute_predictors = nn.ModuleDict()
        
        for category, attributes in category_attributes.items():
            for attr_name in attributes.keys():
                key = f"{category}_{attr_name}"
                if key in attribute_dims:
                    layers = []
                    
                    # Input layer
                    layers.append(nn.Linear(clip_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    
                    # Additional hidden layers
                    for _ in range(num_hidden_layers - 1):
                        layers.append(nn.Linear(hidden_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))
                    
                    # Output layer
                    layers.append(nn.Linear(hidden_dim, attribute_dims[key]))
                    
                    self.attribute_predictors[key] = nn.Sequential(*layers)
    
    def forward(self, clip_features, category):
        results = {}
        category_attrs = self.category_attributes[category]
        
        clip_features = clip_features.float()
        
        for attr_name in category_attrs.keys():
            key = f"{category}_{attr_name}"
            if key in self.attribute_predictors:
                results[key] = self.attribute_predictors[key](clip_features)
        
        return results


def predict_batch(images, categories, clip_model, model, checkpoint, clip_preprocess, device='cuda', batch_size=32):
    """Process a batch of images"""
    all_predictions = []
    
    # Create DataLoader for batch processing
        # Calculate total batches for progress bar
    dataset = ImageDataset(images, categories, clip_preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    from tqdm.auto import tqdm 
    total_batches = len(dataloader)
    with torch.no_grad():
        # Main progress bar for batches
        pbar = tqdm(dataloader, total=total_batches, desc="Processing batches", 
                   unit="batch", position=0, leave=True)
        
        for batch_images, batch_categories in pbar:
            batch_images = batch_images.to(device, non_blocking=True)
            
            # Get CLIP features for the batch
            with torch.autocast('cuda'):
                clip_features = clip_model.encode_image(batch_images)
                
                # Process each category in the batch
                batch_predictions = []
                for idx, category in enumerate(batch_categories):
                    if category not in checkpoint['category_mapping']:
                        batch_predictions.append({})
                        continue
                    
                    # Get model predictions for single image
                    predictions = model(clip_features[idx:idx+1], category)
                    
                    # Convert predictions to attribute values
                    predicted_attributes = {}
                    for key, pred in predictions.items():
                        _, predicted_idx = torch.max(pred, 1)
                        predicted_idx = predicted_idx.item()
                        
                        attr_name = key.split('_', 1)[1]
                        attr_values = checkpoint['attribute_classes'][key]
                        if predicted_idx < len(attr_values):
                            predicted_attributes[attr_name] = attr_values[predicted_idx]
                    
                    batch_predictions.append(predicted_attributes)
            
            all_predictions.extend(batch_predictions)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_predictions

def create_clip_model(device):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'ViT-H-14-quickgelu',
        device=device,
        pretrained="dfn5b",
        precision="fp32",  # Explicitly set precision to fp32
        cache_dir="/scratch/data/m23csa016/meesho_data/"
    )
    # Ensure model is in fp32
    model = model.float()
    return model, preprocess_train

# Option 1: Clean the checkpoint before loading
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' prefix
        name = k.replace('module._orig_mod.', '')
        new_state_dict[name] = v
    return new_state_dict

def load_models(model_path, device):
    """Load both the attribute predictor and CLIP model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    clean_clip_checkpoint = clean_state_dict(checkpoint['swa_clip_model_state_dict'])
    
    # Load CLIP model from checkpoint
    # clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    # Load appropriate CLIP model with explicit fp32 precision
    clip_model, clip_preprocess = create_clip_model(device)
    clip_model.load_state_dict(clean_clip_checkpoint, strict=False)
    
    
    # Initialize model with saved configuration
    model = CategoryAwareAttributePredictor(
        clip_dim=checkpoint['model_config']['clip_dim'],
        category_attributes=checkpoint['dataset_info']['category_mapping'],
        attribute_dims={key: len(values) for key, values in checkpoint['dataset_info']['attribute_classes'].items()},
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        dropout_rate=checkpoint['model_config']['dropout_rate'],
        num_hidden_layers=checkpoint['model_config']['num_hidden_layers']
    ).to(device)
    
    # Load model state
    clean_cat_checkpoint = clean_state_dict(checkpoint['swa_model_state_dict'])
    model.load_state_dict(clean_cat_checkpoint, strict=False)

    # Enable torch.compile if using PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        clip_model = torch.compile(clip_model)
    
    model.eval()
    clip_model.eval()

    return model, clip_model, clip_preprocess, checkpoint['dataset_info']

def process_csv_file(input_csv_path, image_dir, model_path, output_csv_path, batch_size=32, device='cuda'):
    # Load the input CSV
    df = pd.read_csv(input_csv_path)
    
    # Validate required columns
    required_columns = ['id', 'Category']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    # Load models from checkpoint
    model, clip_model, clip_preprocess, checkpoint = load_models(model_path, device)
    
    # Prepare image paths and categories
    image_paths = [
        os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg") 
        for id_ in df['id']
        if os.path.exists(os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg"))
    ]
    valid_indices = [
        i for i, id_ in enumerate(df['id'])
        if os.path.exists(os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg"))
    ]
    categories = df['Category'].iloc[valid_indices].tolist()
    
    print(f"Processing {len(image_paths)} valid images out of {len(df)} total entries")
    
    # Get predictions in batches
    predictions = predict_batch(
        image_paths, 
        categories, 
        clip_model, 
        model, 
        checkpoint, 
        clip_preprocess,
        device=device,
        batch_size=batch_size
    )
    
    # Process results
    results = []
    pred_idx = 0
    for idx, row in df.iterrows():
        if idx in valid_indices:
            pred = predictions[pred_idx]
            pred_idx += 1
        else:
            pred = {}
            
        result = {
            'id': row['id'],
            'Category': row['Category'],
            'len': len(pred)
        }
        
        # Map the predictions to attr_1, attr_2, etc.
        category_mapping = category_class_attribute_mapping[row['Category']]
        
        # Initialize all attribute columns with None
        for i in range(1, 11):
            result[f'attr_{i}'] = "dummy"
            
        # Fill in the predicted attributes according to the mapping
        for attr_name, pred_value in pred.items():
            if attr_name in category_mapping:
                attr_column = category_mapping[attr_name]
                result[attr_column] = pred_value
        
        results.append(result)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    columns = ['id', 'Category', 'len'] + [f'attr_{i}' for i in range(1, 11)]
    output_df = output_df[columns]
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

def main():
    # Configuration
    input_csv_path = "/scratch/data/m23csa016/meesho_data/test.csv"
    image_dir = "/scratch/data/m23csa016/meesho_data/test_images"
    model_path = "/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/vith14_gelu_swa/vith14_gelu_swa_0_trainval_041640.pth"
    output_csv_path = "/iitjhome/m23csa016/meesho_code/results/vith14_gelu_swa_0_trainval_041640.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128 # Adjust based on your GPU memory
    
    # Process the CSV files
    process_csv_file(
        input_csv_path=input_csv_path,
        image_dir=image_dir,
        model_path=model_path,
        output_csv_path=output_csv_path,
        batch_size=batch_size,
        device=device
    )

if __name__ == "__main__":
    main()
