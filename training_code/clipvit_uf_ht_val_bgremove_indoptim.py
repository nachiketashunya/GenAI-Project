import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
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
from torchvision import transforms
from rembg import remove
import timm
from torch.optim.lr_scheduler import MultiStepLR
torch.backends.cudnn.benchmark = True

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

def create_train_val_datasets(dataset, val_ratio=0.05, seed=42):
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

def validate_model(model, clip_model, seres_model, connector_mlp, val_loader, criterion, device, logger):
    """
    Validate the model on validation set
    """
    model.eval()
    clip_model.eval()
    val_loss = 0.0
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    num_batches = 0

    
    with torch.no_grad():
        for images, categories, targets in val_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            clip_features = clip_model.encode_image(images)
            clip_features = clip_features.float()

            seres_features = seres_model(images)
            seres_features = seres_features.float()

            features = torch.cat([clip_features, seres_features], dim=-1)

            if connector_mlp:
                features = connector_mlp(features)
            
            batch_loss = 0
            valid_predictions = 0
            batch_correct = defaultdict(int)
            batch_total = defaultdict(int)
            
            for i in range(batch_size):
                category = categories[i]
                img_features = features[i].unsqueeze(0)
                predictions = model(img_features, category)
                
                for key, pred in predictions.items():
                    target_val = targets[key][i]
                    if target_val != -1:
                        target_tensor = torch.tensor([target_val]).to(device)
                        loss = criterion(pred, target_tensor)
                        batch_loss += loss
                        
                        _, predicted = torch.max(pred, 1)
                        is_correct = (predicted == target_tensor).item()
                        
                        attr_correct[key] += is_correct
                        attr_total[key] += 1

                        batch_correct[key] += is_correct
                        batch_total[key] += 1
                        
                        valid_predictions += 1
            
            if valid_predictions > 0:
                val_loss += (batch_loss / valid_predictions).item()

                num_batches += 1
    
    # Calculate metrics
    avg_val_loss = val_loss / num_batches
    val_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
    
    # Calculate per-attribute accuracies
    attr_accuracies = {
        key: attr_correct[key] / attr_total[key] 
        for key in attr_correct.keys()
        if attr_total[key] > 0
    }
    
    return avg_val_loss, val_acc, attr_accuracies


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

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
    def __init__(self, csv_path, image_dir, train=True,device = 'cuda'):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.train = train
        
        # Load CLIP model and get base preprocess
        self.clip_model, self.base_preprocess = clip.load("ViT-L/14", device= device)
        
        # Define category-specific augmentations
        self.saree_augmentation = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # Match CLIP's resize
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.5),
            transforms.RandomAffine(degrees=15, scale=(0.9, 1.1), shear=5),
            transforms.ToTensor(),  # Convert to tensor after all spatial augmentations
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP's normalization values
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))  # Apply RandomErasing after converting to tensor
        ])
        
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

    def preprocess_image(self, image, category):
        """Apply category-specific preprocessing"""

        if self.train:
            if category == 'Sarees':
                return self.saree_augmentation(image)
            else:
                return self.base_preprocess(image)
        else:
            return self.base_preprocess(image)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        category = row['Category']
        image_path = f"{self.image_dir}/{row['id'].astype(str).zfill(6)}.png"
        
        try:
            # Load and preprocess image with category-specific augmentations
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess_image(image, category)
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
        
class ConnectorMLP(nn.Module):
    def __init__(self, connect_dim=768+2048, clip_dim=768, num_hidden_layers=1, logger=None):
        super(ConnectorMLP, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_hidden_layers-1):
            self.layers.append(nn.Linear(connect_dim, connect_dim//2))
            self.layers.append(nn.ReLU())
            
            connect_dim //= 2
        
        self.layers.append(nn.Linear(connect_dim, clip_dim))
        self.layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*self.layers)

        if logger is not None:
            logger.info(f"\nInitialized Connector MLP with {connect_dim=}, {clip_dim=}, {num_hidden_layers=}")
    
    def forward(self, features):
        return self.model(features)

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

def setup_logging(log_dir="logs_e100"):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'ht_connect_bgremove_{timestamp}.log')
    
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

def train_model_with_validation(model, train_loader, val_loader, device, train_dataset, 
                              clip_lr, predictor_lr, weight_decay, 
                              beta1, beta2, hidden_dim, dropout_rate, num_hidden_layers, 
                              logger, connector_mlp=None, patience=100, num_epochs=10):
    """
    Modified training function with validation and early stopping
    """
    if logger is None:
        logger = setup_logging()
    
    # Log training configuration
    logger.info("Starting new training run with validation")
    log_hyperparameters(logger, {
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

    clip_model, _ = clip.load("ViT-L/14", device=device)
    logger.info("CLIP model loaded successfully")

    seres_model = timm.create_model('seresnextaa101d_32x8d', pretrained=True)
    seres_model = torch.nn.Sequential(*list(seres_model.children())[:-1])
    seres_model = seres_model.to(device)  
    seres_model.eval()
    logger.info("SE-Resnet model loaded successfully")                
    
    optimizer = optim.AdamW([
        {'params': clip_model.parameters(), 'lr': clip_lr},
        {'params':connector_mlp.parameters(),'lr': predictor_lr, 'weight_decay': weight_decay},

        *[{'params': m.parameters(), 'lr': predictor_lr, 'weight_decay': weight_decay} for m in model.attribute_predictors.values()]
    ], betas=(beta1, beta2))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    scheduler = MultiStepLR(optimizer=optimizer, milestones=[40, 70, 80], gamma=0.1)
    
    # Initialize metrics tracking
    patience_counter = 0
    best_val_accuracy = 0
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'params': {
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
                    images = images.to(device)
                    batch_size = images.size(0)
                    
                    clip_features = clip_model.encode_image(images)
                    clip_features = clip_features.float()

                    with torch.no_grad():
                        seres_features = seres_model(images)
                        seres_features = seres_features.float()

                    features = torch.cat([clip_features, seres_features], dim=-1)

                    if connector_mlp is not None:
                        features = connector_mlp(features)
                    
                    batch_loss = 0
                    batch_correct = defaultdict(int)
                    batch_total = defaultdict(int)
                    valid_predictions = 0
                    # attr_loss = defaultdict(list)
                    
                    for i in range(batch_size):
                        category = categories[i]
                        img_features = features[i].unsqueeze(0)
                        predictions = model(img_features, category)
                        
                        for key, pred in predictions.items():
                            target_val = targets[key][i]
                            if target_val != -1:
                                target_tensor = torch.tensor([target_val]).to(device)
                                loss = criterion(pred, target_tensor)
                                
                                # attr_loss[key].append(loss)

                                batch_loss += loss
                                
                                _, predicted = torch.max(pred, 1)
                                is_correct = (predicted == target_tensor).item()
                                
                                attr_correct[key] += is_correct
                                attr_total[key] += 1
                                
                                batch_correct[key] += is_correct
                                batch_total[key] += 1
                                
                                valid_predictions += 1
                    
                    if valid_predictions > 0:
                        optimizer.zero_grad()

                    # for key, val in attr_loss.items():
                    #     if len(val)>0:
                    #         attr_loss[key]=sum(attr_loss[key])/len(attr_loss[key])
                    #         attr_loss[key].backward(retain_graph=True)
                             
                        batch_loss = batch_loss / valid_predictions
                        
                        # Step the optimizer
                        batch_loss.backward()

                        # batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        convert_models_to_fp32(clip_model)
                        optimizer.step()
                        clip.model.convert_weights(clip_model)
                        
                        train_loss += batch_loss.item()
                        num_batches += 1
                        
                        # Log batch metrics
                        batch_metrics['loss'].append(batch_loss.item())
                        batch_acc = sum(batch_correct.values()) / max(sum(batch_total.values()), 1)
                        batch_metrics['accuracy'].append(batch_acc)
                        
                        if batch_idx % 100 == 0:  # Log every 100 batches
                            logger.info(
                                f"Epoch {epoch+1}, Batch {batch_idx}: "
                                f"Loss = {batch_loss.item():.4f}, "
                                f"Accuracy = {batch_acc:.4%}"
                            )
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                    continue

            # Validation phase
            print("\nValidating the model")
            logger.info(
                "\nValidating the model\n"
            )

            val_loss, val_acc, val_attr_accuracies = validate_model(
                model=model, 
                clip_model=clip_model, 
                seres_model=seres_model,
                connector_mlp=connector_mlp,
                val_loader=val_loader, 
                criterion=criterion, 
                device=device, 
                logger=logger
            )

            print("\nValidated")
            logger.info(
                "\nValidated\n"
            )
            
            scheduler.step()

            # Calculate and log epoch metrics
            avg_train_loss = train_loss / num_batches
            train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
            
            metrics_history['train_loss'].append(avg_train_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_acc'].append(val_acc)
            
            
            # Log per-attribute accuracies
            attr_accuracies = {
                key: attr_correct[key] / attr_total[key] 
                for key in attr_correct.keys()
            }

            logger.info(
                f"Epoch {epoch+1} Results:\n"
                f"Train Loss: {avg_train_loss:.4f}\n"
                f"Train Acc: {train_acc:.4%}\n"
                f"Attribute Training Accuracies: {json.dumps(attr_accuracies, indent=2)}"
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
                        'clip_dim': 768,  # If this is variable, make it a parameter
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
                        'checkpoint_version': '1.0'  # Useful for future compatibility
                    }
                }
                
                # Save the checkpoint
                checkpoint_dir = '/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/ht_connect_bgremove_e100'
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                save_path = os.path.join(checkpoint_dir, f'ht_connect_bgremove_indoptim_{epoch+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
                torch.save(checkpoint, save_path)
                
                # Also save a metadata file in JSON format for easy reading
                metadata_path = save_path.replace('.pth', '_metadata.json')
                metadata = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'clip_model_state_dict', 'optimizer_state_dict']}
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"\nSaved best model checkpoint to {save_path}")
                print(f"Saved metadata to {metadata_path}")
            
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
    
    # Hyperparameter grid
    param_grid = {
        'clip_lr': [1e-5],
        'predictor_lr': [5e-5],
        'weight_decay': [0.001],
        'beta1': [0.9],
        'beta2': [0.999],
        'hidden_dim': [256, 512],
        'dropout_rate': [0.1],
        'num_hidden_layers': [1, 2]
    }
    
    logger.info("Hyperparameter search space:")
    logger.info(json.dumps(param_grid, indent=2))
    
    batch_size = 32
    num_epochs = 100
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    try:
        # Data loading setup
        DATA_DIR = "/scratch/data/m23csa016/meesho_data"
        train_csv = os.path.join(DATA_DIR, "train.csv")
        train_images = os.path.join(DATA_DIR, "train_images_bg_removed")
        
        train_dataset = ProductDataset(
            csv_path=train_csv,
            image_dir=train_images,
            train=True,
            device= device
        )
        logger.info(f"Dataset loaded successfully with {len(train_dataset)} samples")

        # Split into train and validation sets
        train_subset, val_subset = create_train_val_datasets(train_dataset)
        logger.info(f"Dataset split: {len(train_subset)} train, {len(val_subset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
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
                clip_dim=768,
                category_attributes=category_class_attribute_mapping,
                attribute_dims=attribute_dims,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                num_hidden_layers=num_hidden_layers
            ).to(device)

            connector_mlp = ConnectorMLP(connect_dim=768+2048, clip_dim=768, num_hidden_layers=1, logger=logger).to(device)
            
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
                        
                    _, _, metrics = train_model_with_validation(
                        model=model,
                        connector_mlp=connector_mlp,
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
                    
                except Exception as e:
                    logger.error(f"Error in training run: {str(e)}", exc_info=True)
                    continue
        
        logger.info("Hyperparameter search completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()