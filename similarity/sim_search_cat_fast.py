import numpy as np
import pandas as pd
from collections import Counter
import faiss
from sklearn.preprocessing import normalize
import os
import json
from datetime import datetime
from tqdm.auto import tqdm
import torch
from typing import List, Dict
import concurrent.futures
from functools import partial

class FAISSIndex:
    def __init__(self, features, filenames):
        self.features = features
        self.filenames = filenames
        self.dimension = features.shape[1]
        self.index = None
        
    def build_index(self):
        # Pre-normalize features
        self.features_normalized = normalize(
            np.ascontiguousarray(self.features.astype('float32'))
        )
        
        num_points = self.features.shape[0]
        
        if num_points < 100000:  # For smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.features_normalized)
        else:
            # For larger datasets, use IVFFlat
            nlist = min(int(np.sqrt(num_points)), num_points // 30)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            print(f"Training index with {nlist} clusters...")
            self.index.train(self.features_normalized)
            self.index.add(self.features_normalized)
            self.index.nprobe = min(nlist // 4, 256)
        
        print("Transferring index to GPU...")
        torch.cuda.empty_cache()
        self.res = faiss.StandardGpuResources()  # GPU resource object
        
        # Transfer index to GPU - simplified call
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        print("Index transferred to GPU.")
        
        return self.index

class SimilaritySearch:
    def __init__(self, index, filenames: List[str], train_df: pd.DataFrame, threshold: float = 0.5, 
                 log_file: str = 'similarity_search_log.jsonl', batch_size: int = 128):
        self.index = index
        self.filenames = filenames
        self.threshold = threshold
        self.log_file = log_file
        self.batch_size = batch_size
        
        # Create category lookup dictionary for faster access
        self.category_lookup = dict(zip(train_df['id_as_filename'], train_df['Category']))
        
        # Pre-compute filename to index mapping
        self.filename_to_idx = {fname: idx for idx, fname in enumerate(filenames)}
        
    @staticmethod
    def _batch_writer(log_file: str, batch_entries: List[Dict]):
        """Write multiple log entries efficiently"""
        with open(log_file, 'a') as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + '\n')
    
    def batch_search(self, query_features: np.ndarray, query_filenames: List[str], k: int = 2048):
        # Normalize query features
        query_features = normalize(query_features.astype('float32'))
        
        # Perform batch search
        distances, indices = self.index.search(query_features, k)
        
        # Process results in batch
        all_results = []
        log_entries = []
        
        for query_filename, dists, idxs in zip(query_filenames, distances, indices):
            query_category = self.category_lookup.get(query_filename)
            if query_category is None:
                continue
            
            # Filter results efficiently using numpy operations
            candidate_filenames = np.array(self.filenames)[idxs]
            candidate_categories = np.array([self.category_lookup.get(f) for f in candidate_filenames])
            
            # Create mask for valid results
            valid_mask = (candidate_categories == query_category) & (dists >= self.threshold)
            
            # Apply mask to get final results
            results = [
                {'filename': str(fname), 'similarity': float(sim)}
                for fname, sim in zip(candidate_filenames[valid_mask], dists[valid_mask])
            ]
            
            # Prepare log entry
            log_entries.append({
                'timestamp': datetime.now().isoformat(),
                'query_filename': query_filename,
                'query_category': query_category,
                'num_similar_images': len(results),
                'similar_images': results
            })
            
            all_results.append((query_filename, results))
        
        # Batch write logs
        if log_entries:
            self._batch_writer(self.log_file, log_entries)
        
        return all_results

def process_category_attributes_batch(similar_images_batch: List[Dict], train_df: pd.DataFrame, 
                                   attribute_columns: List[str]):
    """Process attributes for a batch of similar images"""
    # Create a set of all filenames needed
    all_filenames = {img['filename'] for img in similar_images_batch}
    
    # Filter train_df once for all filenames
    relevant_rows = train_df[train_df['id_as_filename'].isin(all_filenames)]
    
    # Get the actual columns that exist in train_df
    existing_attribute_columns = [col for col in attribute_columns if col in train_df.columns]
    
    # Initialize results with None for all requested attributes
    voted_attributes = {attr: None for attr in attribute_columns}
    
    # Process only existing columns
    for attr in existing_attribute_columns:
        non_null_values = relevant_rows[attr].dropna()
        if len(non_null_values) > 0:
            voted_attributes[attr] = Counter(non_null_values).most_common(1)[0][0]
    
    return voted_attributes

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Create or clear log file
    log_file = 'similarity_search_log.jsonl'
    with open(log_file, 'w') as f:
        f.write('')
    
    # Load data efficiently
    train_df = pd.read_csv(
        '/scratch/data/m23csa016/meesho_data/new_train.csv'
    )
    train_df['id_as_filename'] = train_df['id'].astype(str).str.zfill(6) + '.png'
    
    train_features_df = pd.read_parquet('/scratch/data/m23csa016/meesho_data/cvl_nobg_max_train_em_1.parquet')
    
    feature_cols = [col for col in train_features_df.columns if col.startswith('feature_')]
    train_features = train_features_df[feature_cols].values
    train_filenames = train_features_df['filename'].tolist()
    
    # Build index
    faiss_index = FAISSIndex(train_features, train_filenames)
    index = faiss_index.build_index()
    
    # Filter for category
    working_df = train_df[train_df['Category'] == 'Kurtis'].copy()
    working_df['id_as_filename'] = working_df['id'].astype(str).str.zfill(6) + '.png'
    
    # Initialize attribute columns
    attribute_columns = [f'attr_{i}' for i in range(1, 10)]
    
    # Add missing attribute columns to working_df
    for attr in attribute_columns:
        if attr not in working_df.columns:
            working_df[attr] = None
    
    similarity_search = SimilaritySearch(
        index, 
        train_filenames, 
        train_df, 
        threshold=0.55,
        log_file=log_file,
        batch_size=BATCH_SIZE
    )
    
    # Process in batches with progress tracking
    total_processed = 0
    less_10 = 0
    
    # Create feature lookup dictionary
    feature_lookup = dict(zip(
        train_features_df['filename'],
        [row[feature_cols].values for _, row in train_features_df.iterrows()]
    ))
    
    for batch_start in tqdm(range(0, len(working_df), BATCH_SIZE), desc="Processing Batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(working_df))
        batch_df = working_df.iloc[batch_start:batch_end]
        
        # Get features for batch efficiently
        batch_features = []
        batch_filenames = []
        batch_indices = []
        
        for idx, row in batch_df.iterrows():
            filename = row['id_as_filename']
            if filename in feature_lookup:
                batch_features.append(feature_lookup[filename])
                batch_filenames.append(filename)
                batch_indices.append(idx)
        
        if not batch_features:
            continue
            
        # Perform batch search
        batch_results = similarity_search.batch_search(
            np.array(batch_features), 
            batch_filenames,
            k=500
        )
        
        # Process results
        for (filename, results), idx in zip([r for r in batch_results if len(r[1]) >= 10], batch_indices):
            # Process attributes for single item to avoid batching complexity
            voted_attributes = process_category_attributes_batch(
                results,
                train_df,
                attribute_columns
            )
            
            # Update working_df
            for attr, value in voted_attributes.items():
                working_df.loc[idx, attr] = value
        
        less_10 += sum(1 for _, results in batch_results if len(results) < 10)
        total_processed += len(batch_results)
        
        # Save progress periodically
        if total_processed % (BATCH_SIZE * 10) == 0:
            working_df.to_csv('kurtis_processed_inter.csv', index=False)
            print(f"Saved progress after processing {total_processed} samples.")
            print(f"Total Files less than 10: {less_10}")
    
    # Save final results
    out_csv_dir = "/iitjhome/m23csa016/meesho_code/sim_correct_train/"
    os.makedirs(out_csv_dir, exist_ok=True)
    working_df.to_csv(os.path.join(out_csv_dir, 'kurtis_processed.csv'), index=False)
    print("Processing complete. Final results saved to kurtis_processed.csv")

if __name__ == "__main__":
    main()