import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import clip
from collections import defaultdict
import os
from tqdm.auto import tqdm
from itertools import product
import json
from datetime import datetime
import logging
from collections import defaultdict
from tqdm.auto import tqdm
from itertools import product
import sys
import torch
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import open_clip
import wandb

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

"""
Helper Functions
"""
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def setup_logging(log_dir="logs_e40"):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'vith14_quickgelu_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def log_hyperparameters(logger, params):
    """Log hyperparameters in a structured format"""
    logger.info("Training hyperparameters:")
    logger.info(json.dumps(params, indent=2))


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
    return model, preprocess_train, preprocess_val

def create_train_val_datasets(dataset, val_ratio=0.1, seed=42):
    """
    Split dataset into training and validation sets
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Use random_split to maintain PyTorch Dataset functionality
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset
"""
Helper functions ended
"""

# 1. Replace custom collate_fn with a more efficient version
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    categories = [item[1] for item in batch]
    
    # Pre-allocate targets dictionary with tensor arrays
    targets = {
        f"{category}_{attr_name}": torch.full((len(batch),), -1, dtype=torch.long)
        for category in category_class_attribute_mapping
        for attr_name in category_class_attribute_mapping[category]
    }
    
    # Use vectorized operations instead of loops
    for batch_idx, (_, category, item_targets) in enumerate(batch):
        for key, value in item_targets.items():
            if key in targets:
                targets[key][batch_idx] = value
    
    return images, categories, targets

# 2. Optimize ProductDataset
class ProductDataset(Dataset):
    def __init__(self, csv_path, image_dir, clip_preprocess_train, clip_preprocess_val, train=True, indices=None):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        # Cache file paths
        self.image_paths = [f"{image_dir}/{str(id).zfill(6)}.jpg" for id in self.df['id']]
        
        self.image_dir = image_dir
        self.clip_preprocess_train = clip_preprocess_train
        self.clip_preprocess_val = clip_preprocess_val
        self.train = train
        
        # Pre-compute category attributes and encoders
        self.category_attributes = category_class_attribute_mapping
        self.attribute_encoders = {}
        self.attribute_classes = {}
        
        # Vectorized operations for attribute processing
        for category, attributes in self.category_attributes.items():
            category_mask = self.df['Category'] == category
            category_data = self.df[category_mask]
            
            for attr_name, attr_col in attributes.items():
                mask = (category_data[attr_col].notna()) & (category_data[attr_col] != 'dummy')
                unique_values = category_data.loc[mask, attr_col].unique()
                
                key = f"{category}_{attr_name}"
                self.attribute_classes[key] = list(unique_values)
                self.attribute_encoders[key] = {val: idx for idx, val in enumerate(unique_values)}

        # Pre-compute targets for each sample
        self.cached_targets = self._precompute_targets()
    
    def __len__(self):
        return len(self.df)

    def _precompute_targets(self):
        cached_targets = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            category = row['Category']
            targets = {
                f"{cat}_{attr_name}": -1
                for cat in self.category_attributes
                for attr_name in self.category_attributes[cat]
            }
            
            category_attrs = self.category_attributes[category]
            for attr_name, attr_col in category_attrs.items():
                value = row[attr_col]
                key = f"{category}_{attr_name}"
                
                if pd.notna(value) and value != 'dummy':
                    targets[key] = self.attribute_encoders[key].get(value, -1)
            
            cached_targets.append(targets)
        return cached_targets

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.clip_preprocess_train(image) if self.train else self.clip_preprocess_val(image)
            return image, self.df.iloc[idx]['Category'], self.cached_targets[idx]
        except Exception as e:
            return torch.zeros((3, 336, 336)), self.df.iloc[idx]['Category'], self.cached_targets[idx]

# 3. Optimize validation function
def validate_model(model, clip_model, val_loader, criterion, device, logger):
    model.eval()
    clip_model.eval()
    val_loss = 0.0
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    num_batches = 0
    
    with torch.no_grad(), torch.autocast("cuda"):  # Enable automatic mixed precision
        for images, categories, targets in val_loader:
            images = images.to(device)
            clip_features = clip_model.encode_image(images)
            
            # Process by category in parallel
            category_predictions = {}
            for category in set(categories):
                indices = [i for i, c in enumerate(categories) if c == category]
                if not indices:
                    continue
                    
                category_features = clip_features[indices]
                predictions = model(category_features, category)
                category_predictions[category] = (indices, predictions)
            
            batch_loss = 0
            valid_predictions = 0
            
            for category, (indices, predictions) in category_predictions.items():
                for key, pred in predictions.items():
                    target_vals = targets[key][indices]
                    valid_mask = target_vals != -1
                    
                    if valid_mask.any():
                        target_tensor = target_vals[valid_mask].to(device)
                        loss = criterion(pred[valid_mask], target_tensor)
                        batch_loss += loss
                        
                        _, predicted = torch.max(pred[valid_mask], 1)
                        correct = (predicted == target_tensor).sum().item()
                        
                        attr_correct[key] += correct
                        attr_total[key] += valid_mask.sum().item()
                        valid_predictions += valid_mask.sum().item()
            
            if valid_predictions > 0:
                val_loss += (batch_loss / valid_predictions).item()
                num_batches += 1

    avg_val_loss = val_loss / max(num_batches, 1)
    val_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
    
    attr_accuracies = {
        key: attr_correct[key] / attr_total[key] 
        for key in attr_correct.keys()
        if attr_total[key] > 0
    }
    
    return avg_val_loss, val_acc, attr_accuracies

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
                    layers.append(nn.ReLU())
                    
                    # Additional hidden layers
                    for _ in range(num_hidden_layers - 1):
                        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))

                        hidden_dim //= 2
                    
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



def train_model_with_validation(clip_model, model, train_loader, val_loader, device, train_dataset, 
                              clip_lr, predictor_lr, weight_decay, 
                              beta1, beta2, hidden_dim, dropout_rate, num_hidden_layers, 
                              logger, patience=30, num_epochs=10):
    """
    Modified training function with validation, early stopping, and support for multiple CLIP models
    """
    if logger is None:
        logger = setup_logging()
    
    # Log training configuration
    logger.info("Starting new training run with validation")
    log_hyperparameters(logger, {
        'clip_model': "laion",
        'clip_lr': clip_lr,
        'predictor_lr': predictor_lr,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'num_epochs': num_epochs,
        'patience': patience,
        'device': str(device),
        'model_architecture': str(model)
    })
    
    # Ensure the attribute predictor is also in fp32
    model = model.float()
    
    optimizer = optim.AdamW([
        {'params': clip_model.parameters(), 'lr': clip_lr},
        {'params': model.parameters(), 'lr': predictor_lr, 'weight_decay': weight_decay}
    ], betas=(beta1, beta2))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # Enable automatic mixed precision training
    scaler = torch.GradScaler("cuda")

    # Initialize metrics tracking
    patience_counter = 0
    best_val_accuracy = 0
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'params': {
            'clip_model': "laion",
            'clip_lr': clip_lr,
            'predictor_lr': predictor_lr,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'num_hidden_layers': num_hidden_layers
        }
    }
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    try:
        for epoch in epoch_pbar:
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            clip_model.train()
            train_loss = 0.0
            attr_correct = defaultdict(int)
            attr_total = defaultdict(int)
            
            train_pbar = tqdm(train_loader, 
                            desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                            leave=False, 
                            position=1)
            
            num_batches = 0
            batch_metrics = defaultdict(list)

            for batch_idx, (images, categories, targets) in enumerate(train_pbar):
                try:
                    # Ensure images are in fp32
                    images = images.to(device).float()
                    batch_size = images.size(0)
                    
                    # Use automatic mixed precision
                    with torch.autocast("cuda"):
                        clip_features = clip_model.encode_image(images)
                        clip_features = clip_features.float()
                        
                        batch_loss = 0
                        batch_correct = defaultdict(int)
                        batch_total = defaultdict(int)
                        valid_predictions = 0
                        
                        # Group samples by category
                        category_indices = defaultdict(list)
                        for idx, category in enumerate(categories):
                            category_indices[category].append(idx)
                        
                        # Process each category group
                        batch_loss = 0
                        valid_predictions = 0
                        
                        for category, indices in category_indices.items():
                            # Get features for this category
                            category_features = clip_features[indices]
                            
                            # Get predictions for this category group
                            predictions = model(category_features, category)
                            
                            # Process predictions for each attribute in this category
                            for key, pred in predictions.items():
                                target_vals = targets[key][indices]
                                valid_mask = target_vals != -1
                                valid_count = valid_mask.sum().item()
                                
                                if valid_count > 0:
                                    target_tensor = target_vals[valid_mask].to(device)
                                    loss = criterion(pred[valid_mask], target_tensor)
                                    batch_loss += loss
                                    
                                    _, predicted = torch.max(pred[valid_mask], 1)
                                    correct = (predicted == target_tensor).sum().item()
                                    
                                    attr_correct[key] += correct
                                    attr_total[key] += valid_count

                                    batch_correct[key] += correct
                                    batch_total[key] += valid_count

                                    valid_predictions += valid_count
                    
                    if valid_predictions > 0:
                        batch_loss = batch_loss / valid_predictions

                        optimizer.zero_grad()
                        # Use gradient scaling
                        scaler.scale(batch_loss).backward()
                        scaler.unscale_(optimizer)
                     
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Handle mixed precision for OpenAI CLIP
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        train_loss += batch_loss.item()
                        num_batches += 1
                        
                        # Log batch metrics
                        batch_metrics['loss'].append(batch_loss.item())
                        batch_acc = sum(batch_correct.values()) / max(sum(batch_total.values()), 1)
                        batch_metrics['accuracy'].append(batch_acc)
                        
                        if batch_idx % 100 == 0:
                            logger.info(
                                f"Epoch {epoch+1}, Batch {batch_idx}: "
                                f"Loss = {batch_loss.item():.4f}, "
                                f"Accuracy = {batch_acc:.4%}"
                            )
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                    continue

            # Validation phase
            val_loss, val_acc, val_attr_accuracies = validate_model(
                model, clip_model, val_loader, criterion, device, logger
            )
            
            # Calculate and log epoch metrics
            avg_train_loss = train_loss / max(num_batches, 1)
            train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
            
            metrics_history['train_loss'].append(avg_train_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_acc'].append(val_acc)

            wandb.log({
                'Train/Loss': avg_train_loss,
                'Train/Acc': train_acc,
                'Val/Loss': val_loss,
                'Val/Acc': val_acc
            })
            
            # Log per-attribute accuracies
            attr_accuracies = {
                key: attr_correct[key] / attr_total[key] 
                for key in attr_correct.keys()
            }

            logger.info(
                f"Epoch {epoch+1} Results:\n"
                f"Train Loss: {avg_train_loss:.4f}\n"
                f"Train Acc: {train_acc:.4%}\n"
                f"Attribute Training Accuracies: {json.dumps(attr_accuracies, indent=2)}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Val Acc: {val_acc:.4%}\n"
                f"Attribute Validation Accuracies: {json.dumps(val_attr_accuracies, indent=2)}"
            )
            
            # Update best accuracy and save comprehensive checkpoint
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                
                checkpoint = {
                    # Model states
                    'model_state_dict': model.state_dict(),
                    'clip_model_state_dict': clip_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    
                    # Model architecture parameters
                    'model_config': {
                        'clip_model': "laion",
                        'clip_dim': 1024,
                        'hidden_dim': hidden_dim,
                        'dropout_rate': dropout_rate,
                        'num_hidden_layers': num_hidden_layers,
                    },
                    
                    # Training parameters
                    'training_config': {
                        'clip_lr': clip_lr,
                        'predictor_lr': predictor_lr,
                        'weight_decay': weight_decay,
                        'beta1': beta1,
                        'beta2': beta2,
                        'batch_size': train_loader.batch_size,
                        'num_epochs': num_epochs,
                    },
                    
                    # Dataset information
                    'dataset_info': {
                        'attribute_classes': train_dataset.attribute_classes,
                        'attribute_encoders': train_dataset.attribute_encoders,
                        'category_mapping': category_class_attribute_mapping,
                    },
                    
                    # Training metrics
                    'metrics': {
                        'best_val_accuracy': best_val_accuracy,
                        'final_epoch': epoch,
                        'training_history': metrics_history,
                        'training_attr_accs': attr_accuracies,
                        'validation_attr_accs': val_attr_accuracies
                    },
                    
                    # Date and version info
                    'metadata': {
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'checkpoint_version': '1.0'
                    }
                }
                
                # Save checkpoints
                checkpoint_dir = '/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/vith14_quickgelu'
                os.makedirs(checkpoint_dir, exist_ok=True)

                save_path = os.path.join(checkpoint_dir, 
                    f'vith14_quickgelu_{epoch}_trainval_{datetime.now().strftime("%H%M%S")}.pth')
                torch.save(checkpoint, save_path)
                
                # Save metadata
                metadata_path = save_path.replace('.pth', '_metadata.json')
                metadata = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'clip_model_state_dict', 'optimizer_state_dict']}
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(
                    f"Saved best model checkpoint to {save_path}\n"
                    f"Saved metadata to {metadata_path}"
                )
                
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    except Exception as e:
        logger.error(f"Training interrupted: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training completed successfully")
    return model, clip_model, metrics_history

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting hyperparameter search")

    # Log into wandb
    wandb.login(key="82fadbf5b2810c5fdaee488a728eabb8f084b7a3")
    logger.info("WandB login successful!")
    
    # Hyperparameter grid
    param_grid = {
        'clip_lr': [1e-6, 1e-4, 5e-6],
        'predictor_lr': [5e-5, 5e-4, 1e-3],
        'weight_decay': [0.001, 0.05, 0.1],
        'beta1': [0.9, 0.95],
        'beta2': [0.999, 0.9999],
        'hidden_dim': [512, 256, 384],
        'dropout_rate': [0.1, 0.3, 0.4, 0.5],
        'num_hidden_layers': [2, 3]
    }
    
    logger.info("Hyperparameter search space:")
    logger.info(json.dumps(param_grid, indent=2))

    
    batch_size = 16
    num_epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clip_model, clip_preprocess_train, clip_preprocess_val = create_clip_model(device=device)
    logger.info("LAION CLIP model loaded successfully in fp32 precision")
    
    try:
        # Data loading setup
        DATA_DIR = "/scratch/data/m23csa016/meesho_data"
        train_images = os.path.join(DATA_DIR, "train_images")
        train_csv = os.path.join(DATA_DIR, "train.csv")
        df = pd.read_csv(train_csv)

        # Create stratification labels
        # You can use category or any other column you want to stratify on
        stratify_labels = df['Category']  # or multiple columns combined if needed

        # Split indices with stratification
        train_indices, val_indices = train_test_split(
            np.arange(len(df)),
            test_size=0.1,
            random_state=42,
            stratify=stratify_labels  # This ensures proportional splitting
        )

        # Create datasets
        train_dataset = ProductDataset(
            csv_path=train_csv,
            image_dir=train_images,
            clip_preprocess_train=clip_preprocess_train,
            clip_preprocess_val=clip_preprocess_val,
            train=True,
            indices=train_indices
        )

        val_dataset = ProductDataset(
            csv_path=train_csv,
            image_dir=train_images,
            clip_preprocess_train=clip_preprocess_train,
            clip_preprocess_val=clip_preprocess_val,
            train=False,
            indices=val_indices
        )

        logger.info(f"Train Dataset: {len(train_dataset)} samples\nValidation Dataset: {len(val_dataset)}")

        # Split into train and validation sets
        # train_subset, val_subset = create_train_val_datasets(train_dataset)
        # logger.info(f"Dataset split: {len(train_subset)} train, {len(val_subset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        attribute_dims = {
            key: len(values) 
            for key, values in train_dataset.attribute_classes.items()
        }
        
        logger.info("Attribute dimensions:")
        logger.info(json.dumps(attribute_dims, indent=2))
        
        all_results = []
        
        # Systematic hyperparameter search
        for hidden_dim, dropout_rate, num_hidden_layers in product(
            param_grid['hidden_dim'],
            param_grid['dropout_rate'],
            param_grid['num_hidden_layers']
        ):
            model = CategoryAwareAttributePredictor(
                clip_dim=1024,
                category_attributes=category_class_attribute_mapping,
                attribute_dims=attribute_dims,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                num_hidden_layers=num_hidden_layers
            ).to(device)

            # Enable torch.compile if using PyTorch 2.0+
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
                clip_model = torch.compile(clip_model)
            
            logger.info(f"\nInitialized model with architecture parameters:")
            logger.info(f"Hidden Dim: {hidden_dim}, Dropout: {dropout_rate}, Layers: {num_hidden_layers}")
            
            for clip_lr, predictor_lr, weight_decay, beta1, beta2 in product(
                param_grid['clip_lr'],
                param_grid['predictor_lr'],
                param_grid['weight_decay'],
                param_grid['beta1'],
                param_grid['beta2']
            ):
                try:
                    logger.info(f"\nStarting training with optimizer parameters:")
                    logger.info(
                        f"CLIP LR: {clip_lr}, Predictor LR: {predictor_lr}, "
                        f"Weight Decay: {weight_decay}, Beta1: {beta1}, Beta2: {beta2}"
                    )

                    config = {
                        "hidden_dim": hidden_dim,
                        "num_hidden_layers": num_hidden_layers,
                        "clip_lr": clip_lr,
                        "predictor_lr": predictor_lr,
                        "weight_decay": weight_decay,
                        "beta1": beta1,
                        "beta2": beta2
                    }

                    run_name = f"h{hidden_dim}_n{num_hidden_layers}"
                    wandb.init(
                        project="ViT-H-14-Quickgelu",
                        name=run_name,
                        config=config
                    )
                        
                    _, _, metrics = train_model_with_validation(
                        clip_model=clip_model,
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        train_dataset=train_dataset,
                        clip_lr=clip_lr,
                        predictor_lr=predictor_lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        hidden_dim=hidden_dim,
                        dropout_rate=dropout_rate,
                        num_hidden_layers=num_hidden_layers,
                        num_epochs=num_epochs,
                        logger=logger
                    )
                    
                    result = {
                        'hidden_dim': hidden_dim,
                        'dropout_rate': dropout_rate,
                        'num_hidden_layers': num_hidden_layers,
                        'clip_lr': clip_lr,
                        'predictor_lr': predictor_lr,
                        'weight_decay': weight_decay,
                        'beta1': beta1,
                        'beta2': beta2,
                        'final_accuracy': metrics['train_acc'][-1],
                        'best_accuracy': max(metrics['train_acc']),
                        'final_loss': metrics['train_loss'][-1],
                        'best_loss': min(metrics['train_loss'])
                    }
                    
                    all_results.append(result)
                    logger.info("Training run completed successfully")
                    logger.info(f"Results: {json.dumps(result, indent=2)}")
                    
                    # Save results after each run
                    with open('hyperparameter_search_results.json', 'w') as f:
                        json.dump(all_results, f, indent=2)
                    logger.info("Updated hyperparameter search results saved to file")

                    wandb.log(result)
                    wandb.finish()
                    
                except Exception as e:
                    logger.error(f"Error in training run: {str(e)}", exc_info=True)
                    continue
        
        logger.info("Hyperparameter search completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()