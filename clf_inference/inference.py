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

class CategoryAwareAttributePredictor(nn.Module):
    def __init__(self, clip_dim=512, category_attributes=None, attribute_dims=None):
        super(CategoryAwareAttributePredictor, self).__init__()
        
        self.category_attributes = category_attributes
        
        # Create prediction heads for each category-attribute combination
        self.attribute_predictors = nn.ModuleDict()
        
        self.dropout_rate = 0.2
        self.l2_lambda = 0.01
        self.hidden_dims = [512]
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(clip_dim)

        for category, attributes in category_attributes.items():
            for attr_name in attributes.keys():
                key = f"{category}_{attr_name}"
                if key in attribute_dims:
                    layers = []
                    
                    # Input layer with normalization and dropout
                    in_dim = clip_dim
                    for hidden_dim in self.hidden_dims:
                        layers.extend([
                            nn.Linear(in_dim, hidden_dim),
                            # nn.BatchNorm1d(hidden_dim),  # Batch normalization
                            nn.ReLU(),
                            nn.Dropout(self.dropout_rate)
                        ])

                        in_dim = hidden_dim
                    
                    # Output layer
                    layers.append(nn.Linear(in_dim, attribute_dims[key]))
                    
                    self.attribute_predictors[key] = nn.Sequential(*layers)

    def forward(self, clip_features, category, training=False):
        # results = {}
        # category_attrs = self.category_attributes[category]

        # Convert to float and apply input normalization
        clip_features = self.input_norm(clip_features.float())
        # clip_features = clip_features.float()
        
        results = {}
        regularization_loss = 0.0
        category_attrs = self.category_attributes[category]
        
        # Enable/disable dropout based on training mode
        self.train(training)
        
        for attr_name in category_attrs.keys():
            key = f"{category}_{attr_name}"
            if key in self.attribute_predictors:
                results[key] = self.attribute_predictors[key](clip_features)

                # Calculate L2 regularization loss during training
                if training:
                    for param in self.attribute_predictors[key].parameters():
                        regularization_loss += torch.norm(param, p=2)
        
        # Scale regularization loss
        regularization_loss *= self.l2_lambda
        
        return results, regularization_loss
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0
        for key in self.attribute_predictors:
            for param in self.attribute_predictors[key].parameters():
                l1_loss += torch.abs(param).sum()
        return l1_loss

    def apply_weight_decay(self, weight_decay=0.01):
        """Apply manual weight decay to all parameters"""
        with torch.no_grad():
            for key in self.attribute_predictors:
                for param in self.attribute_predictors[key].parameters():
                    param.mul_(1 - weight_decay)

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

def load_models(model_path, device):
    """Load both the attribute predictor and CLIP model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load CLIP model from checkpoint
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.load_state_dict(checkpoint['clip_model_state_dict'])
    clip_model.eval()
    
    
    # Load attribute predictor
    attribute_dims = {
        key: len(values) 
        for key, values in checkpoint['attribute_classes'].items()
    }
    
    model = CategoryAwareAttributePredictor(
        clip_dim=512,
        category_attributes=checkpoint['category_mapping'],
        attribute_dims=attribute_dims
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, clip_model, clip_preprocess, checkpoint

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
            clip_features = clip_model.encode_image(batch_images)
            
            # Process each category in the batch
            batch_predictions = []
            for idx, category in enumerate(batch_categories):
                if category not in checkpoint['category_mapping']:
                    batch_predictions.append({})
                    continue
                
                # Get model predictions for single image
                predictions, _ = model(clip_features[idx:idx+1], category)
                
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
    model_path = "/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_base/cv_uf_enhanced_70k_45.pth"
    output_csv_path = "cv_uf_enhanced_70k_45.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128  # Adjust based on your GPU memory
    
    # Process the CSV file
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
