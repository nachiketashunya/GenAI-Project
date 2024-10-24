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
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        
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
        self.hidden_dims = [512, 256]
        
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
                    
                    # Weight initialization
                    self._initialize_weights(self.attribute_predictors[key])
    
    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, clip_features, category, training=True):
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

def validate_model(model, clip_model, val_loader, device, criterion):
    model.eval()
    clip_model.eval()
    val_loss = 0.0
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, categories, targets in val_pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            clip_features = clip_model.encode_image(images)
            clip_features = clip_features.float()
            
            total_loss = 0
            batch_attr_correct = defaultdict(int)
            
            for i in range(batch_size):
                category = categories[i]
                img_features = clip_features[i].unsqueeze(0)
                predictions, reg_loss = model(img_features, category, training=False)
                
                for key, pred in predictions.items():
                    target_val = targets[key][i]
                    if target_val != -1:
                        target_tensor = torch.tensor([target_val]).to(device)
                        loss = criterion(pred, target_tensor, key)
                        total_loss += loss
                        
                        _, predicted = torch.max(pred, 1)
                        is_correct = (predicted == target_tensor).item()
                        attr_correct[key] += is_correct
                        attr_total[key] += 1
                        batch_attr_correct[key] += is_correct
            
            if total_loss > 0:
                total_loss = total_loss / batch_size
                val_loss += total_loss.item()
            
            val_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_acc': f'{sum(batch_attr_correct.values()) / max(sum(attr_total.values()), 1):.2%}'
            })
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
    
    return avg_val_loss, val_acc, attr_correct, attr_total


def analyze_class_distribution(dataset):
    """
    Analyzes class distribution for each category-attribute combination
    Returns detailed statistics and imbalance metrics
    """
    class_stats = defaultdict(lambda: defaultdict(int))
    category_stats = defaultdict(int)
    
    # Count occurrences
    for _, category, targets in dataset:
        category_stats[category] += 1
        for key, value in targets.items():
            if value != -1:  # Ignore missing values
                class_stats[key][value] += 1
    
    return class_stats, category_stats

def compute_balanced_weights(class_stats):
    """
    Computes balanced weights for each category-attribute combination
    using effective number of samples to handle extreme imbalance
    """
    weights = {}
    beta = 0.9999  # Hyperparameter for effective number of samples
    
    for attr_key, class_counts in class_stats.items():
        if not class_counts:
            continue
            
        # Convert counts to tensor for numerical stability
        counts = torch.tensor(list(class_counts.values()), dtype=torch.float)
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, counts)
        weights_per_class = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights_per_class = weights_per_class / weights_per_class.sum()
        
        # Store weights with class indices
        weights[attr_key] = {
            class_idx: weight.item() 
            for class_idx, weight in enumerate(weights_per_class)
        }
    
    return weights

class WeightedFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with class weights and attribute-specific handling
    """
    def __init__(self, class_weights, alpha=1, gamma=2, reduction='mean', ignore_index=-1):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, attr_key):
        """
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            attr_key: The category_attribute key to get correct weights
        """
        # Get weights for this specific attribute
        if attr_key in self.class_weights:
            weights = torch.tensor(
                [self.class_weights[attr_key].get(i, 1.0) for i in range(inputs.size(-1))],
                device=inputs.device
            )
        else:
            weights = None

        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=weights,
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss[targets != self.ignore_index].mean()
        elif self.reduction == 'sum':
            return focal_loss[targets != self.ignore_index].sum()
        return focal_loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    Create a schedule with linear warmup and cosine decay
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, device, train_dataset, num_epochs=10):
    clip_model, _ = clip.load("ViT-B/32", device=device)

    print("Model Loaded")
    filename = "class_weights.json"

    if os.path.exists(filename):
        with open(filename, 'r') as json_file:
            class_weights = json.load(json_file)
    else:
        # Analyze class distribution
        class_stats, category_stats = analyze_class_distribution(train_dataset)
        class_weights = compute_balanced_weights(class_stats)
    
        with open(filename, 'w') as json_file:
            json.dump(class_weights, json_file, indent=4)

    # Initialize weighted loss
    criterion = WeightedFocalLoss(
        class_weights=class_weights,
        alpha=1,
        gamma=2,
        ignore_index=-1
    )

    # 1. Simplified parameter groups with proper weight decay
    predictor_params = model.parameters()
    clip_params = clip_model.parameters()
    
    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': 5e-5, 'weight_decay': 0.01},
        {'params': predictor_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])

    # Add warmup to scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    
    # 3. Add label smoothing to loss function for better generalization
    # criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    # Focal Loss for handling class imbalance
    criterion = WeightedFocalLoss(class_weights=class_weights, alpha=1, gamma=2, ignore_index=-1)
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        clip_model.train()
        train_loss = 0.0
        attr_correct = defaultdict(int)
        attr_total = defaultdict(int)
        
        train_pbar = tqdm(train_loader, 
                         desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, 
                         position=1)
        
        for images, categories, targets in train_pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            # 4. Add gradient scaling for mixed precision training
            clip_features = clip_model.encode_image(images)
            clip_features = clip_features.float()
            
            total_loss = 0
            batch_attr_correct = defaultdict(int)
            
            for i in range(batch_size):
                category = categories[i]
                img_features = clip_features[i].unsqueeze(0)
                predictions, reg_loss = model(img_features, category, training=True)

                for key, pred in predictions.items():
                    target_val = targets[key][i]
                    if target_val != -1:
                        target_tensor = torch.tensor([target_val]).to(device)
                        # Pass attribute key to loss function
                        loss = criterion(pred, target_tensor, key)
                        # loss = criterion(pred, target_tensor)
                        total_loss += loss
                        
                        _, predicted = torch.max(pred, 1)
                        is_correct = (predicted == target_tensor).item()
                        attr_correct[key] += is_correct
                        attr_total[key] += 1
                        batch_attr_correct[key] += is_correct

            # Add regularization
            if total_loss > 0:
                total_loss = total_loss / batch_size + reg_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                convert_models_to_fp32(clip_model)
                optimizer.step()
                scheduler.step()
                clip.model.convert_weights(clip_model)
            
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_acc': f'{sum(batch_attr_correct.values()) / max(sum(attr_total.values()), 1):.2%}'
            })
    

        # Validation phase
        avg_val_loss, val_acc, val_attr_correct, val_attr_total = validate_model(
            model, clip_model, val_loader, device, criterion
        )
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
        
        # # Early stopping check
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     # Save best model
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'clip_model_state_dict': clip_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'epoch': epoch,
        #         'attribute_classes': train_dataset.attribute_classes,
        #         'attribute_encoders': train_dataset.attribute_encoders,
        #         'category_mapping': category_class_attribute_mapping,
        #         'best_val_loss': best_val_loss
        #     }, f'clipvit_unfreeze_70k_best.pth')
        
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
        }, f'/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_base/cv_uf_enhanced_70k_{epoch}.pth')

        print(f'\nCheckpoint saved')
        print('-' * 80)
    
    return model, clip_model

def main():
    batch_size = 32
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default dtype for torch operations
    # torch.set_default_dtype(torch.float32)
    
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

import pandas as pd
import os
from tqdm import tqdm
import torch
from PIL import Image
import clip
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    model_path = "/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_base/cv_uf_enhanced_70k_0.pth"
    output_csv_path = "cv_base_uf_enhanced_70k_0.csv"
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
