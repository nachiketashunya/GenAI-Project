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

class FAISSIndex:
    def __init__(self, features, filenames):
        self.features = features
        self.filenames = filenames
        self.dimension = features.shape[1]
        self.index = None
        
    def build_index(self):
        features_c = np.ascontiguousarray(self.features.astype('float32'))
        features_c = normalize(features_c)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(features_c)

        print("Transferring index to GPU...")
        torch.cuda.empty_cache()
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        print("Index transferred to GPU.")
        
        return self.index

class SimilaritySearch:
    def __init__(self, index, filenames, train_df, threshold=0.5, log_file='similarity_search_log.jsonl'):
        self.index = index
        self.filenames = filenames
        self.train_df = train_df
        self.threshold = threshold
        self.log_file = log_file
        
    def log_search_results(self, query_filename, results, category):
        """Log search results to a JSONL file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_filename': query_filename,
            'query_category': category,
            'num_similar_images': len(results),
            'similar_images': results
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def search_with_category_constraint(self, query_feature, query_filename, k=2048):
        query_feature = normalize(query_feature.astype('float32').reshape(1, -1))
        distances, indices = self.index.search(query_feature, k)

        query_category = self.train_df[
            self.train_df['id_as_filename'] == query_filename
        ]['Category'].values[0]

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = float(dist)
            candidate_filename = self.filenames[idx]
            
            candidate_category = self.train_df[
                self.train_df['id_as_filename'] == candidate_filename
            ]['Category'].values[0]
            
            if (candidate_category == query_category and 
                similarity >= self.threshold):
                results.append({
                    'filename': candidate_filename,
                    'similarity': similarity
                })
        
        self.log_search_results(query_filename, results, query_category)
        
        return results

def process_category_attributes(similar_images, train_df):
    length = 9 # Fixed length for Men Tshirts
    similar_train_rows = train_df[
        train_df['id_as_filename'].isin([img['filename'] for img in similar_images])
    ]
    
    attribute_columns = [f'attr_{i}' for i in range(1, length + 1)]
    
    voted_attributes = {}
    for attr in attribute_columns:
        non_null_values = similar_train_rows[attr].dropna()
        
        if len(non_null_values) > 0:
            most_common = Counter(non_null_values).most_common(1)[0][0]
            voted_attributes[attr] = most_common
        else:
            voted_attributes[attr] = None
    
    return voted_attributes

def main():
    # Create or clear log file
    log_file = 'similarity_search_log.jsonl'
    with open(log_file, 'w') as f:
        f.write('')
    
    # Load data
    train_df = pd.read_csv('/scratch/data/m23csa016/meesho_data/new_train.csv')
    train_df['id_as_filename'] = train_df['id'].astype(str).str.zfill(6) + '.png'
    
    train_features_df = pd.read_parquet('/scratch/data/m23csa016/meesho_data/cvl_nobg_max_train_em_1.parquet')
    
    feature_cols = [col for col in train_features_df.columns if col.startswith('feature_')]
    train_features = train_features_df[feature_cols].values
    train_filenames = train_features_df['filename'].tolist()
    
    faiss_index = FAISSIndex(train_features, train_filenames)
    index = faiss_index.build_index()

    # Filter for only Men Tshirts
    working_df = train_df[train_df['Category'] == 'Kurtis'].copy()
    working_df['id_as_filename'] = working_df['id'].astype(str).str.zfill(6) + '.png'
    
    # Initialize attribute columns
    for i in range(1, 10):  # 5 attributes for Men Tshirts
        working_df[f'attr_{i}'] = None
    
    similarity_search = SimilaritySearch(
        index, 
        train_filenames, 
        train_df, 
        threshold=0.55,  # Modified threshold
        log_file=log_file
    )
    
    dump_interval = 200
    total_processed = 0
    less_10 = 0


    for idx, row in tqdm(working_df.iterrows(), total=len(working_df), desc="Processing Rows"):
        sample_filename = f"{row['id_as_filename']}"
        
        sample_feature_row = train_features_df[train_features_df['filename'] == sample_filename]
        
        if len(sample_feature_row) > 0:
            sample_feature = sample_feature_row[feature_cols].values[0]
            
            similar_images = similarity_search.search_with_category_constraint(
                sample_feature, sample_filename, k=500
            )
            
            if len(similar_images) >= 10:
                voted_attributes = process_category_attributes(
                    similar_images, 
                    train_df
                )
                
                for attr, value in voted_attributes.items():
                    working_df.loc[idx, attr] = value
                
                # print(f"Processed {sample_filename}: {len(similar_images)} similar images")
            else:
                # print(f"Skipped {sample_filename}: only {len(similar_images)} similar images")
                less_10 += 1
        else:
            print(f"Could not find feature for {sample_filename}")
            
        total_processed += 1
        
        if total_processed % dump_interval == 0:
            working_df.to_csv('men_tshirts_processed.csv', index=False)
            print(f"Saved progress after processing {total_processed} samples.")
            print(f"Total Files less than 10: {less_10}")
    
    out_csv_dir = "/iitjhome/m23csa016/meesho_code/sim_correct_train/"
    os.makedirs(out_csv_dir, exist_ok=True)

    working_df.to_csv(os.path.join(out_csv_dir, 'kurtis_processed.csv'), index=False)
    print("Processing complete. Final results saved to kurtis_processed.csv")

if __name__ == "__main__":
    main()