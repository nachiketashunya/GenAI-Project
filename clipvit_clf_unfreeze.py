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

        dropout_rate=0.2
        
        for category, attributes in category_attributes.items():
            for attr_name in attributes.keys():
                key = f"{category}_{attr_name}"
                if key in attribute_dims:
                    self.attribute_predictors[key] = nn.Sequential(
                        nn.Linear(clip_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, attribute_dims[key])
                    )
    
    def forward(self, clip_features, category):
        results = {}
        category_attrs = self.category_attributes[category]

        clip_features = clip_features.float()
        
        for attr_name in category_attrs.keys():
            key = f"{category}_{attr_name}"
            if key in self.attribute_predictors:
                results[key] = self.attribute_predictors[key](clip_features)
        
        return results
    

def train_model(model, train_loader, device, train_dataset, num_epochs=10):
    clip_model, _ = clip.load("ViT-L/14", device=device)

    # Option 1: Set requires_grad=False for all parameters
    # for param in clip_model.parameters():
    #     param.requires_grad = False
        # Unfreeze last few transformer layers
    # visual_blocks = clip_model.visual.transformer.resblocks
    # num_blocks = len(visual_blocks)
    # blocks_to_finetune = int(num_blocks * 0.25)
    
    # for i in range(num_blocks - blocks_to_finetune, num_blocks):
    #     for param in visual_blocks[i].parameters():
    #         param.requires_grad = True
    #     # Create parameter groups with different learning rates
    
    # clip_params = [p for p in clip_model.parameters() if p.requires_grad]
    predictor_params = model.parameters()
    clip_params = clip_model.parameters()
    
    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': 5e-5},
        {'params': predictor_params, 'lr': 1e-4, "weight_decay": 0.01}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        clip_model.train()
        train_loss = 0.0
        attr_correct = defaultdict(int)
        attr_total = defaultdict(int)
        
        # Batch progress bar for training
        train_pbar = tqdm(train_loader, 
                         desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, 
                         position=1)
        
        for images, categories, targets in train_pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            clip_features = clip_model.encode_image(images)
            clip_features = clip_features.float()
            
            total_loss = 0
            batch_attr_correct = defaultdict(int)
            
            for i in range(batch_size):
                category = categories[i]
                img_features = clip_features[i].unsqueeze(0)
                predictions = model(img_features, category)
                # print(f"{predictions=}")

                for key, pred in predictions.items():
                    target_val = targets[key][i]
                    if target_val != -1:
                        target_tensor = torch.tensor([target_val]).to(device)
                        loss = criterion(pred, target_tensor)
                        total_loss += loss
                        
                        _, predicted = torch.max(pred, 1)
                        is_correct = (predicted == target_tensor).item()
                        attr_correct[key] += is_correct
                        attr_total[key] += 1
                        batch_attr_correct[key] += is_correct
            
            if total_loss > 0:
                total_loss = total_loss/batch_size
                optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping (optional, but can help with training stability)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                           # Convert model's parameters to FP32 format, update, and convert back
                convert_models_to_fp32(clip_model)
                optimizer.step()
                clip.model.convert_weights(clip_model)
                train_loss += total_loss.item()
            
            # Update training progress bar with current batch metrics
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_acc': f'{sum(batch_attr_correct.values()) / max(sum(attr_total.values()), 1):.2%}'
            })
        
       
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        
        # Calculate average accuracies
        train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
        
        # Update main progress bar with epoch metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_acc': f'{train_acc:.2%}'
        })
        
        # Print detailed metrics for each attribute
        print(f'\nEpoch {epoch+1}/{num_epochs} Detailed Metrics:')
        for key in attr_total.keys():
            if attr_total[key] > 0:
                train_acc = 100 * attr_correct[key] / attr_total[key]
                print(f'{key}: Train Acc: {train_acc:.2f}%')

        
        # Save the trained model and mappings
        torch.save({
            'model_state_dict': model.state_dict(),
            'clip_model_state_dict':clip_model.state_dict(),
            'attribute_classes': train_dataset.attribute_classes,
            'attribute_encoders': train_dataset.attribute_encoders,
            'category_mapping': category_class_attribute_mapping
        }, f'/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/clipvit_unfreeze_70k_{epoch}.pth')

        print(f'\nNew model saved')

        print('-' * 80)
    
    return model, clip_model

def predict_attributes(model, image_path, category, dataset, device):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = dataset.clip_preprocess(image)
    image = image.unsqueeze(0).to(device)
    
    # Get CLIP features
    with torch.no_grad():
        clip_features = dataset.clip_model.encode_image(image)
        predictions = model(clip_features, category)
    
    # Convert predictions to attribute values
    predicted_attributes = {}
    for key, pred in predictions.items():
        _, predicted_idx = torch.max(pred, 1)
        predicted_idx = predicted_idx.item()
        
        # Get attribute name from key (category_attribute)
        attr_name = key.split('_', 1)[1]
        
        # Convert index back to attribute value
        attr_values = dataset.attribute_classes[key]
        if predicted_idx < len(attr_values):
            predicted_attributes[attr_name] = attr_values[predicted_idx]
    
    return predicted_attributes

def main():
    batch_size = 32
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default dtype for torch operations
    # torch.set_default_dtype(torch.float32)
    
    DATA_DIR = "/scratch/data/m23csa016/meesho_data"
    train_csv = os.path.join(DATA_DIR, "train.csv")
    train_images = os.path.join(DATA_DIR, "train_images")

    # val_csv = os.path.join(DATA_DIR, "clipvit_val.csv")
    # val_images = os.path.join(DATA_DIR, "train_images")

    # Initialize dataset
    train_dataset = ProductDataset(
        csv_path=train_csv,
        image_dir=train_images,
        train=True
    )
    
    # val_dataset = ProductDataset(
    #     csv_path=val_csv,
    #     image_dir=val_images,
    #     train=False
    # )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Add this line
    )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=custom_collate_fn  # Add this line
    # )
    
    # Get number of classes for each category-attribute combination
    attribute_dims = {
        key: len(values) 
        for key, values in train_dataset.attribute_classes.items()
    }
    
    # Initialize model
    model = CategoryAwareAttributePredictor(
        clip_dim=768,
        category_attributes=category_class_attribute_mapping,
        attribute_dims=attribute_dims
    ).to(device)
    
    # Train model
    model,clip_model = train_model(model, train_loader, device, train_dataset, num_epochs)
    
    # Save the trained model and mappings
    torch.save({
        'model_state_dict': model.state_dict(),
        'clip_model_state_dict':clip_model.state_dict(),
        'attribute_classes': train_dataset.attribute_classes,
        'attribute_encoders': train_dataset.attribute_encoders,
        'category_mapping': category_class_attribute_mapping
    }, '/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/clipvit_unfreeze_70k.pth')

if __name__ == "__main__":
    main()