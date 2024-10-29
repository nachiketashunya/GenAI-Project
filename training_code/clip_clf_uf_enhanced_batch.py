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

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

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

# Custom collate function to handle different categories and their attributes
def custom_collate_fn(batch):
    # Separate images, categories, and targets
    images = torch.stack([item[0] for item in batch])
    categories = [item[1] for item in batch]
    
    # Initialize an empty targets dict with all possible category-attribute combinations
    targets = {}
    for category, attrs in category_class_attribute_mapping.items():
        for attr_name in attrs.keys():
            key = f"{category}_{attr_name}"
            targets[key] = torch.full((len(batch),), -1, dtype=torch.long)
    
    # Fill in the actual values
    for batch_idx, (_, category, item_targets) in enumerate(batch):
        for key, value in item_targets.items():
            if key in targets:
                targets[key][batch_idx] = value
    
    return images, categories, targets

class ProductDataset(Dataset):
    def __init__(self, csv_path, image_dir, train=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.train = train
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda")
        
        # Store category-wise attribute information
        self.category_attributes = category_class_attribute_mapping
        
        # Create attribute encoders and store unique values for each attribute
        self.attribute_encoders = {}
        self.attribute_classes = {}
        
        # Initialize all possible category-attribute combinations
        for category, attributes in self.category_attributes.items():
            category_data = self.df[self.df['Category'] == category]
            
            for attr_name, attr_col in attributes.items():
                # Get unique values excluding NULL/dummy values
                unique_values = category_data[attr_col][
                    (category_data[attr_col].notna()) & 
                    (category_data[attr_col] != 'dummy')
                ].unique()
                
                key = f"{category}_{attr_name}"
                self.attribute_classes[key] = list(unique_values)
                self.attribute_encoders[key] = {
                    value: idx for idx, value in enumerate(unique_values)
                }
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        category = row['Category']
        image_path = f"{self.image_dir}/{row['id'].astype(str).zfill(6)}.jpg"
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image)
            
            # Initialize targets with all possible category-attribute combinations
            targets = {}
            for cat, attrs in self.category_attributes.items():
                for attr_name, attr_col in attrs.items():
                    key = f"{cat}_{attr_name}"
                    # Default value is -1 (ignore index)
                    targets[key] = -1
            
            # Fill in the actual values for this category
            category_attrs = self.category_attributes[category]
            for attr_name, attr_col in category_attrs.items():
                value = row[attr_col]
                key = f"{category}_{attr_name}"
                
                if pd.isna(value) or value == 'dummy':
                    targets[key] = -1
                else:
                    targets[key] = self.attribute_encoders[key].get(value, -1)
            
            return image, category, targets
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return default values with all possible category-attribute combinations
            targets = {
                f"{cat}_{attr}": -1 
                for cat in self.category_attributes
                for attr in self.category_attributes[cat].keys()
            }
            return torch.zeros((3, 224, 224)), category, targets

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

    def forward(self, clip_features, categories, training=True):
        clip_features = self.input_norm(clip_features.float())
        
        # Initialize results dictionary for each category-attribute combination
        results = {}
        regularization_loss = 0.0
        
        # Group samples by category
        category_indices = defaultdict(list)
        for idx, category in enumerate(categories):
            category_indices[category].append(idx)
            
        # Process each category group
        for category, indices in category_indices.items():
            category_features = clip_features[indices]
            category_attrs = self.category_attributes[category]
            
            # Process each attribute for this category
            for attr_name in category_attrs.keys():
                key = f"{category}_{attr_name}"
                if key in self.attribute_predictors:
                    # Get predictions for all samples of this category
                    batch_predictions = self.attribute_predictors[key](category_features)
                    
                    # Initialize tensor for all samples if not exists
                    if key not in results:
                        results[key] = torch.full(
                            (len(categories), batch_predictions.size(-1)), 
                            float('-inf'), 
                            device=clip_features.device
                        )
                    
                    # Fill in predictions for this category's samples
                    results[key][indices] = batch_predictions

        return results

def validate_model(model, clip_model, val_loader, device, criterion):
    model.eval()
    clip_model.eval()
    val_loss = 0.0
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    num_batches = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for batch_idx, (images, categories, targets) in enumerate(val_pbar):
            images = images.to(device)
            
            # Get CLIP features for entire batch
            clip_features = clip_model.encode_image(images)
            
            # Get predictions for entire batch
            predictions = model(clip_features, categories, training=False)
            
            # Calculate batch metrics
            batch_loss = 0.0
            valid_predictions = 0
            batch_correct = defaultdict(int)
            batch_total = defaultdict(int)
            
            # Process predictions for each attribute
            for key in predictions:
                target_vals = targets[key].to(device)
                mask = target_vals != -1
                
                if mask.any():
                    # Only compute metrics for valid targets
                    valid_predictions += mask.sum().item()
                    pred = predictions[key][mask]
                    target = target_vals[mask]
                    
                    # Calculate loss
                    loss = criterion(pred, target)
                    batch_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(pred, 1)
                    correct = (predicted == target).sum().item()
                    
                    # Update counters
                    batch_correct[key] += correct
                    batch_total[key] += len(target)
                    
                    # Update global counters
                    attr_correct[key] += correct
                    attr_total[key] += len(target)
            
            # Calculate batch metrics if we have valid samples
            if valid_predictions > 0:
                total_loss = batch_loss / valid_predictions + reg_loss
                val_loss += total_loss
                num_batches += 1
            
            # Calculate batch accuracy for progress bar
            batch_acc = sum(batch_correct.values()) / max(sum(batch_total.values()), 1)
            
            # Update progress bar
            val_pbar.set_postfix({
                'loss': f'{total_loss:.4f}' if valid_predictions > 0 else 'N/A',
                'batch_acc': f'{batch_acc:.2%}',
                'processed': f'{batch_idx+1}/{len(val_loader)}'
            })
    
    # Calculate final metrics
    avg_val_loss = val_loss / max(num_batches, 1)
    val_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
    
    # Print detailed validation metrics
    print("\nDetailed Validation Metrics:")
    for key in sorted(attr_total.keys()):
        if attr_total[key] > 0:
            acc = 100 * attr_correct[key] / attr_total[key]
            print(f'{key}: Acc: {acc:.2f}% ({attr_correct[key]}/{attr_total[key]})')
    
    return avg_val_loss, val_acc, attr_correct, attr_total

def train_model(model, train_loader, val_loader, device, train_dataset, num_epochs=10):
    clip_model, _ = clip.load("ViT-L/14", device=device)
    print("Model Loaded")
    
    predictor_params = model.parameters()
    clip_params = clip_model.parameters()
    
    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': 5e-5, 'weight_decay': 0.01},
        {'params': predictor_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        clip_model.train()
        train_loss = 0.0
        attr_correct = defaultdict(int)
        attr_total = defaultdict(int)
        num_batches = 0
        
        train_pbar = tqdm(train_loader, 
                         desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, 
                         position=1)
        
        for batch_idx, (images, categories, targets) in enumerate(train_pbar):
            images = images.to(device)
            
            # Get CLIP features for entire batch
            clip_features = clip_model.encode_image(images)
            
            # Get predictions for entire batch
            predictions = model(clip_features, categories, training=True)
            
            # Calculate loss across all predictions
            batch_loss = 0.0
            valid_predictions = 0
            
            for key in predictions:
                target_vals = targets[key].to(device)
                mask = target_vals != -1
                
                if mask.any():
                    # Only compute loss for valid targets
                    valid_predictions += mask.sum().item()
                    pred = predictions[key][mask]
                    target = target_vals[mask]
                    
                    loss = criterion(pred, target)
                    batch_loss += loss
                    
                    # Calculate accuracy
                    _, predicted = torch.max(pred, 1)
                    attr_correct[key] += (predicted == target).sum().item()
                    attr_total[key] += len(target)
            
            # Normalize loss by number of valid predictions
            if valid_predictions > 0:
                batch_loss = batch_loss / valid_predictions
                
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                convert_models_to_fp32(clip_model)
                optimizer.step()
                clip.model.convert_weights(clip_model)
                
                train_loss += batch_loss.item()
                num_batches += 1
            
            # Update progress bar
            batch_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
            train_pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}' if valid_predictions > 0 else 'N/A',
                'batch_acc': f'{batch_acc:.2%}'
            })

        # Step the scheduler once per epoch
        scheduler.step()

        # Calculate epoch metrics
        avg_train_loss = train_loss / max(num_batches, 1)
        train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)

        # Validation phase
        avg_val_loss, val_acc, val_attr_correct, val_attr_total = validate_model(
            model, clip_model, val_loader, device, criterion
        )
        
        # Print epoch metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_acc': f'{train_acc:.2%}',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_acc': f'{val_acc:.2%}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        print(f'\nEpoch {epoch+1}/{num_epochs} Detailed Metrics:')
        print('Training Metrics:')
        for key in attr_total.keys():
            if attr_total[key] > 0:
                acc = 100 * attr_correct[key] / attr_total[key]
                print(f'{key}: Train Acc: {acc:.2f}%')
        
        print('\nValidation Metrics:')
        for key in val_attr_total.keys():
            if val_attr_total[key] > 0:
                acc = 100 * val_attr_correct[key] / val_attr_total[key]
                print(f'{key}: Val Acc: {acc:.2f}%')
        
        if epoch % 5 == 0:
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'clip_model_state_dict': clip_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'attribute_classes': train_dataset.attribute_classes,
                'attribute_encoders': train_dataset.attribute_encoders,
                'category_mapping': category_class_attribute_mapping
            }, f'/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large_enhanced/cv_uf_enhanced_70k_{epoch}.pth')

            print(f'\nCheckpoint saved')
            print('-' * 80)
    
    return model, clip_model

def main():
    batch_size = 64
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATA_DIR = "/scratch/data/m23csa016/meesho_data"
    train_csv = os.path.join(DATA_DIR, "clipvit_train_56k.csv")
    train_images = os.path.join(DATA_DIR, "train_images")

    val_csv = os.path.join(DATA_DIR, "clipvit_val_14k.csv")
    val_images = os.path.join(DATA_DIR, "train_images")

    # Initialize dataset
    train_dataset = ProductDataset(
        csv_path=train_csv,
        image_dir=train_images,
        train=True
    )
    
    val_dataset = ProductDataset(
        csv_path=val_csv,
        image_dir=val_images,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add this line
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add this line
    )
    
    # Get number of classes for each category-attribute combination
    attribute_dims = {
        key: len(values) 
        for key, values in train_dataset.attribute_classes.items()
    }
    
    # Initialize model
    model = CategoryAwareAttributePredictor(
        clip_dim=512,
        category_attributes=category_class_attribute_mapping,
        attribute_dims=attribute_dims
    ).to(device)
    
    # Train model
    model,clip_model = train_model(model, train_loader, val_loader, device, train_dataset, num_epochs)

if __name__ == "__main__":
    main()