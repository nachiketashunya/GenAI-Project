import os
from pathlib import Path
import gc
from tqdm import tqdm
import psutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import clip
from typing import List, Tuple


def save_embeddings_to_parquet(embeddings, filenames, output_file, compression='zstd', chunk_size=5000):
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix != '.parquet':
            output_path = output_path.with_suffix('.parquet')
        
        # Check if embeddings and filenames have the same length
        if len(embeddings) != len(filenames):
            raise ValueError(f"Mismatch in lengths: embeddings ({len(embeddings)}) and filenames ({len(filenames)})")
        
        # Remove file extensions and rename column to file_id
        file_ids = [Path(f).stem for f in filenames]
        
        mem_available = psutil.virtual_memory().available
        estimated_memory = embeddings.nbytes * 2  # Factor of 2 for safety margin
        
        print(f"Available memory: {mem_available / 1e9:.2f} GB")
        print(f"Estimated memory needed: {estimated_memory / 1e9:.2f} GB")
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Number of file_ids: {len(file_ids)}")
        
        if estimated_memory > mem_available * 0.7:  # Use 70% of available memory as threshold
            print("Processing in chunks due to memory constraints...")
            
            feature_cols = [f'feature_{i}' for i in range(embeddings.shape[1])]
            schema = pa.schema([
                ('file_id', pa.string()),  # Changed column name to file_id
                *[(col, pa.float32()) for col in feature_cols]
            ])
            
            writer = pq.ParquetWriter(
                output_path,
                schema,
                compression=compression,
                row_group_size=chunk_size,
                use_dictionary=False,
                write_statistics=True
            )
            
            for start_idx in tqdm(range(0, len(file_ids), chunk_size), desc="Saving chunks"):
                end_idx = min(start_idx + chunk_size, len(file_ids))
                
                chunk_embeddings = embeddings[start_idx:end_idx]
                chunk_file_ids = file_ids[start_idx:end_idx]
                
                chunk_dict = {'file_id': chunk_file_ids}  # Updated to file_id
                for i in range(chunk_embeddings.shape[1]):
                    chunk_dict[f'feature_{i}'] = chunk_embeddings[:, i].astype('float32')
                
                chunk_df = pd.DataFrame(chunk_dict)
                chunk_table = pa.Table.from_pandas(chunk_df, schema=schema)
                
                writer.write_table(chunk_table)
                
                del chunk_embeddings, chunk_file_ids, chunk_dict, chunk_df, chunk_table
                gc.collect()
            
            writer.close()
            
        else:
            print("Processing entire dataset at once...")
            feature_cols = [f'feature_{i}' for i in range(embeddings.shape[1])]
            
            data_dict = {'file_id': file_ids}  # Updated to file_id
            for i in range(embeddings.shape[1]):
                data_dict[f'feature_{i}'] = embeddings[:, i].astype('float32')
            
            df = pd.DataFrame(data_dict)
            df['file_id'] = df['file_id'].astype('string[pyarrow]')
            
            table = pa.Table.from_pandas(df)
            
            pq.write_table(
                table,
                output_path,
                compression=compression,
                row_group_size=chunk_size,
                use_dictionary=False,
                write_statistics=True
            )
        
        file_size = output_path.stat().st_size
        print(f"\nEmbeddings saved to {output_path}")
        print(f"File size: {file_size / 1e9:.2f} GB")
        print(f"Number of features: {embeddings.shape[1]}")
        print(f"Number of images: {embeddings.shape[0]}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error saving embeddings to parquet: {str(e)}")
        raise


class CLIPEmbedder:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the CLIP embedder with a custom checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint containing CLIP model weights
            device: Device to run the model on ('cuda' or 'cpu'). If None, automatically detected.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        # Load the base CLIP model and preprocessor
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # Load custom weights if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Check if the checkpoint contains the expected key
            if 'clip_model_state_dict' not in checkpoint:
                raise ValueError("Checkpoint does not contain 'clip_model_state_dict'")
            self.model.load_state_dict(checkpoint['clip_model_state_dict'])
        
        self.model.eval()
        
    @torch.no_grad()
    def process_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Process a batch of PIL images and return their embeddings.
        
        Args:
            images: List of PIL images in RGB format
            
        Returns:
            Tensor of image embeddings
        """
        # Process all images in the batch at once
        pixel_values = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        return self.model.encode_image(pixel_values)
    
    def embed_folder(self, folder_path: str, batch_size: int = 128) -> Tuple[np.ndarray, List[str]]:
        """
        Process all images in a folder and return their embeddings.
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Number of images to process at once
            
        Returns:
            Tuple of (embeddings array, list of image filenames)
        """
        # Get all image files
        image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        if not image_files:
            raise ValueError(f"No valid image files found in {folder_path}")
        
        embeddings = []
        valid_files = []
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            
            # Load and validate images
            for img_file in batch_files:
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    valid_files.append(img_file)
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
                    continue
            
            if batch_images:
                # Get embeddings for valid images
                batch_embeddings = self.process_batch(batch_images)
                embeddings.extend(batch_embeddings.cpu().numpy())
                
                # Clean up
                for img in batch_images:
                    img.close()
        
        if not embeddings:
            raise ValueError("No valid embeddings were generated")
            
        return np.array(embeddings), valid_files

def main():
    # Example usage
    folder_path = "/scratch/data/m23csa016/meesho_data/train_images"
    checkpoint_path = "/scratch/data/m23csa016/meesho_data/checkpoints/clipvit_large/hyper_tune/cvl_uf_ht_2_70k.pth"
    output_file = "/scratch/data/m23csa016/meesho_data/embeddings/cvl_uf_ht_2_70k_train_em.parquet"
    
    # Initialize embedder
    embedder = CLIPEmbedder(checkpoint_path)
    
    # # Generate embeddings
    embeddings, image_files = embedder.embed_folder(folder_path)
    
    # Save embeddings
    save_embeddings_to_parquet(embeddings, image_files, output_file)
    
    # Process test images
    test_folder = "/scratch/data/m23csa016/meesho_data/test_images"
    test_output = "/scratch/data/m23csa016/meesho_data/embeddings/cvl_uf_ht_2_70k_test_em.parquet"
    
    test_embeddings, test_files = embedder.embed_folder(test_folder)
    save_embeddings_to_parquet(test_embeddings, test_files, test_output)

if __name__ == "__main__":
    main()