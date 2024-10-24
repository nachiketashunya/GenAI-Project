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
        # Normalize features for cosine similarity (using inner product)
        features_c = np.ascontiguousarray(self.features.astype('float32'))
        features_c = normalize(features_c)  # L2 normalization

        # Create FAISS index with inner product distance
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(features_c)

        print("Transferring index to GPU...")
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        print("Index transferred to GPU.")

class SimilaritySearch:
    def __init__(self, index, filenames, threshold=0.7):
        self.index = index
        self.filenames = filenames
        self.threshold = threshold
        
    def search(self, query_features, k=100):
        # Normalize query features for cosine similarity
        query_features = normalize(query_features.astype('float32'))

        # Search the index for the top k nearest neighbors
        distances, indices = self.index.search(query_features, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = dist  # Cosine similarity is directly the distance in inner product space
            if similarity >= self.threshold:  # Check threshold for similarity
                results.append({
                    'filename': self.filenames[idx],
                    'similarity': similarity
                })
        
        return results

def process_single_image(image_path, similarity_search, feature_extractor):
    """Process a single image and find similar images."""
    # Extract features for the single image
    features = feature_extractor.extract_single_feature(image_path)
    
    # Perform similarity search
    similar_images = similarity_search.search(features.reshape(1, -1))
    
    # Create DataFrame with results
    results = {
        'test_image': os.path.basename(image_path),
        'similar_images': similar_images
    }
    
    # Expand similar images into separate columns
    expanded = {'test_image': results['test_image']}
    for i, img in enumerate(results['similar_images'], 1):
        expanded[f'similar_image_{i}'] = img['filename']
        expanded[f'similarity_{i}'] = img['similarity']
    
    return pd.DataFrame([expanded])

class OptimizedCategoryBasedSimilaritySearch:
    def __init__(self, similarity_search, filenames, category_mapping, train_df):
        self.similarity_search = similarity_search
        self.filenames = np.array(filenames)
        self.category_mapping = category_mapping
        self.train_df = train_df
        
        # Precompute categories for all filenames
        self.filename_categories = self.precompute_categories()
        print(f"Initialized OptimizedCategoryBasedSimilaritySearch with {len(self.filename_categories)} category mappings")
    
    def precompute_categories(self):
        if isinstance(self.category_mapping, pd.DataFrame):
            return pd.Series(self.category_mapping.set_index('id')['Category']).to_dict()
        else:
            return self.category_mapping
    
    def get_category(self, filename):
        category = self.filename_categories.get(filename, 'default')
        return category
    
    def search_batch(self, query_features, query_filenames, k=200):
        print(f"Searching batch of {len(query_filenames)} items")
        results = []
        for query_feature, query_filename in zip(query_features, query_filenames):
            similar_images = self.similarity_search.search(query_feature.reshape(1, -1), k=k)
            category = self.get_category(query_filename)
            filtered_results = [
                img for img in similar_images
            ][:k]  # Ensure we don't exceed k results after filtering
            
            results.append({
                'query_filename': query_filename,
                'similar_filenames': [img['filename'] for img in filtered_results],
                'similarities': [img['similarity'] for img in filtered_results],
                'category': category
            })
        
        return results

def process_batch(batch, similarity_search, test_categories_df, train_df, k=200):
    batch_features, batch_filenames = zip(*batch)
    batch_features = np.array(batch_features)
    
    print(f"Processing batch of {len(batch_filenames)} items")
    search_results = similarity_search.search_batch(batch_features, batch_filenames, k=k)
    
    len_category_mapping = {
        'Kurtis': 9,
        'Men Tshirts': 5,
        'Sarees': 10,
        'Women Tops & Tunics': 10,
        'Women Tshirts': 8
    }
    
    results = []
    for result in search_results:
        train_df['id_as_filename'] = train_df['id'].astype(str).str.zfill(6) + '.jpg'
        
        # Get the category of the test sample
        test_categories_df['id_as_filename'] = test_categories_df['id'].astype(str).str.zfill(6) + '.jpg'
        matching_rows = test_categories_df[test_categories_df['id_as_filename'] == result['query_filename']]
        if len(matching_rows) > 0:
            test_row = matching_rows.iloc[0]
            category = test_row['Category']
            length = len_category_mapping[category]
        else:
            category = 'Unknown'
            length = 0
        
        # Filter similar images to only include those from the same category
        similar_train_rows = train_df[
            (train_df['id_as_filename'].isin(result['similar_filenames'])) &
            (train_df['Category'] == category)
        ].head(k)
        
        attribute_columns = [f'attr_{i}' for i in range(1, length + 1)] if length else []
        
        voted_attributes = {}
        for attr in attribute_columns:
            non_null_values = similar_train_rows[attr].dropna()
            if len(non_null_values) > 0:
                voted_attributes[attr] = Counter(non_null_values).most_common(1)[0][0]
            else:
                voted_attributes[attr] = None
        
        # Fill remaining attributes with None
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
    batch_size = 128 # Set batch size
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

# Load necessary data
train_df = pd.read_csv('/scratch/data/m23csa016/meesho_data/train.csv')
test_categories_df = pd.read_csv('/scratch/data/m23csa016/meesho_data/test.csv')
print(f"Loaded train data: {len(train_df)} rows, test categories: {len(test_categories_df)} rows")

# Load pre-computed embeddings
train_features_df = pd.read_parquet('clipvit_train_embeddings_1.parquet')
test_features_df = pd.read_parquet('clipvit_test_embeddings_1.parquet')

feature_cols = [col for col in train_features_df.columns if col.startswith('feature_')]
train_features = train_features_df[feature_cols].values
train_filenames = train_features_df['filename'].tolist()

# Extract feature columns
feature_cols = [col for col in test_features_df.columns if col.startswith('feature_')]
test_features = test_features_df[feature_cols].values

# Extract and sort filenames based on the numeric part
test_filenames = test_features_df['filename'].tolist()

# Sort the filenames and associated features based on the numeric part of 'filename'
sorted_indices = sorted(range(len(test_filenames)), key=lambda i: int(test_filenames[i].split('.')[0].zfill(6)))
test_filenames = [test_filenames[i] for i in sorted_indices]
test_features = test_features[sorted_indices]

print("Building FAISS index")
# Build FAISS index
faiss_index = FAISSIndex(train_features, train_filenames)
faiss_index.build_index()

# Initialize similarity search
similarity_search = SimilaritySearch(faiss_index.index, train_filenames, threshold=0)

print("Initializing optimized category-based similarity search")
# Initialize the optimized category-based similarity search
optimized_similarity = OptimizedCategoryBasedSimilaritySearch(
    similarity_search=similarity_search,
    filenames=train_filenames,
    category_mapping=test_categories_df,
    train_df=train_df
)

print("Starting to process all images")
# Process all images with category-specific matching and attribute voting
results_df = process_images_with_categories(
    test_features,
    test_filenames,
    test_categories_df,
    optimized_similarity,
    train_df,
    k=15
)

# Save the results to submission.csv
submission_columns = ['id', 'Category', 'len', 'attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 
                     'attr_6', 'attr_7', 'attr_8', 'attr_9', 'attr_10']
results_df[submission_columns].to_csv('clipvit_K15_results.csv', index=False)
print("Results saved to submission.csv")