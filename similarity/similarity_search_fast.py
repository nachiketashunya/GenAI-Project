import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import faiss
import os
from sklearn.preprocessing import normalize

class FAISSIndex:
    def __init__(self, features, filenames):
        self.features = features
        self.filenames = filenames
        self.dimension = features.shape[1]
        self.index = None
        
    def build_index(self):
        # Normalize features (preserved from original)
        features_c = np.ascontiguousarray(self.features.astype('float32'))
        features_c = normalize(features_c)

        # Create GPU index directly instead of CPU->GPU transfer
        self.res = faiss.StandardGpuResources()
        gpu_config = faiss.GpuIndexFlatConfig()
        gpu_config.useFloat16 = True  # Memory optimization
        self.index = faiss.GpuIndexFlatIP(self.res, self.dimension, gpu_config)
        self.index.add(features_c)
        print("Index built on GPU directly.")

def process_batch(batch, similarity_search, test_categories_df, train_df, k=200):
    batch_features, batch_filenames = zip(*batch)
    batch_features = np.array(batch_features)
    
    print(f"Processing batch of {len(batch_filenames)} items")
    
    # Pre-compute filename mappings once per batch
    train_df['id_as_filename'] = train_df['id'].astype(str).str.zfill(6) + '.jpg'
    test_categories_df['id_as_filename'] = test_categories_df['id'].astype(str).str.zfill(6) + '.jpg'
    
    # Original length mapping preserved
    len_category_mapping = {
        'Kurtis': 9,
        'Men Tshirts': 5,
        'Sarees': 10,
        'Women Tops & Tunics': 10,
        'Women Tshirts': 8
    }
    
    search_results = similarity_search.search_batch(batch_features, batch_filenames, k=k)
    
    results = []
    for result in search_results:
        # Get category of test sample (preserved logic)
        matching_rows = test_categories_df[test_categories_df['id_as_filename'] == result['query_filename']]
        if len(matching_rows) > 0:
            test_row = matching_rows.iloc[0]
            category = test_row['Category']
            length = len_category_mapping[category]
        else:
            category = 'Unknown'
            length = 0
        
        # Filter similar images by category (preserved logic)
        similar_train_rows = train_df[
            (train_df['id_as_filename'].isin(result['similar_filenames'])) &
            (train_df['Category'] == category)
        ].head(k)
        
        # Attribute voting (preserved logic)
        attribute_columns = [f'attr_{i}' for i in range(1, length + 1)] if length else []
        voted_attributes = {}
        
        # Process attributes in chunks for memory efficiency
        chunk_size = 50  # Process 50 attributes at a time
        for i in range(0, len(attribute_columns), chunk_size):
            chunk = attribute_columns[i:i + chunk_size]
            for attr in chunk:
                non_null_values = similar_train_rows[attr].dropna()
                if len(non_null_values) > 0:
                    voted_attributes[attr] = Counter(non_null_values).most_common(1)[0][0]
                else:
                    voted_attributes[attr] = None
        
        # Fill remaining attributes (preserved logic)
        for i in range(len(attribute_columns) + 1, 11):
            voted_attributes[f'attr_{i}'] = "dummy"
        
        results.append({
            'id': result['query_filename'],
            'Category': category,
            'len': length,
            **voted_attributes
        })
    
    return results

def process_images_with_categories(test_features, test_filenames, test_categories_df, similarity_search, train_df, k=200):
    batch_size = 64  # Slightly increased but still conservative
    total_batches = len(test_filenames) // batch_size + (1 if len(test_filenames) % batch_size > 0 else 0)
    
    results = []
    with tqdm(total=len(test_filenames), desc="Processing images") as pbar:
        for i in range(0, len(test_filenames), batch_size):
            batch = list(zip(test_features[i:i+batch_size], test_filenames[i:i+batch_size]))
            print(f"Processing batch {i//batch_size + 1}/{total_batches}")
            batch_results = process_batch(batch, similarity_search, test_categories_df, train_df, k=k)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    print(f"Processed all {len(test_filenames)} images")
    return pd.DataFrame(results)

class OptimizedCategoryBasedSimilaritySearch:
    def __init__(self, similarity_search, filenames, category_mapping, train_df):
        self.similarity_search = similarity_search
        self.filenames = np.array(filenames)
        self.category_mapping = category_mapping
        self.train_df = train_df
        self.filename_categories = self.precompute_categories()
        print(f"Initialized OptimizedCategoryBasedSimilaritySearch with {len(self.filename_categories)} category mappings")
    
    def precompute_categories(self):
        if isinstance(self.category_mapping, pd.DataFrame):
            return pd.Series(self.category_mapping.set_index('id')['Category']).to_dict()
        else:
            return self.category_mapping
    
    def get_category(self, filename):
        return self.filename_categories.get(filename, 'default')
    
    def search_batch(self, query_features, query_filenames, k=200):
        print(f"Searching batch of {len(query_filenames)} items")
        results = []
        
        # Process features in smaller chunks to manage memory
        chunk_size = 32
        for i in range(0, len(query_features), chunk_size):
            chunk_features = query_features[i:i+chunk_size]
            chunk_filenames = query_filenames[i:i+chunk_size]
            
            for query_feature, query_filename in zip(chunk_features, chunk_filenames):
                similar_images = self.similarity_search.search(query_feature.reshape(1, -1), k=k)
                category = self.get_category(query_filename)
                filtered_results = [img for img in similar_images][:k]
                
                results.append({
                    'query_filename': query_filename,
                    'similar_filenames': [img['filename'] for img in filtered_results],
                    'similarities': [img['similarity'] for img in filtered_results],
                    'category': category
                })
        
        return results
