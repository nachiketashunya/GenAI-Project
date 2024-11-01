import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
import os
import torch
from PIL import Image
import clip
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

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

class CLIPEmbedder:
    def __init__(self, checkpoint_path: str = None, device: str = None):
        """
        Initialize the CLIP embedder with an optional custom checkpoint.
        
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
            if 'clip_model_state_dict' not in checkpoint:
                raise ValueError("Checkpoint does not contain 'clip_model_state_dict'")
            self.model.load_state_dict(checkpoint['clip_model_state_dict'])
        
        self.model.eval()
    
    @torch.no_grad()
    def extract_single_feature(self, image_path: str) -> np.ndarray:
        """Extract features from a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(pixel_values)
            image.close()
            return features.cpu().numpy()
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")

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
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        print("Index transferred to GPU.")

class SimilaritySearch:
    def __init__(self, index, filenames, train_categories, threshold=0.75):
        self.index = index
        self.filenames = filenames
        self.train_categories = train_categories  # Add train categories
        self.threshold = threshold
        
    def search(self, query_features, query_category, k=10):
        """
        Search for similar images within the same category
        
        Args:
            query_features: Features of the query image
            query_category: Category of the query image
            k: Number of results to return
        """
        # Get more initial results to ensure enough after category filtering
        initial_k = k * 3
        query_features = normalize(query_features.astype('float32'))
        distances, indices = self.index.search(query_features, initial_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = dist
            filename = self.filenames[idx]
            # Only include results from the same category
            if (similarity >= self.threshold and 
                self.train_categories.get(filename) == query_category):
                results.append({
                    'filename': filename,
                    'similarity': float(similarity),
                    'category': query_category
                })
        
        # Return top k results after category filtering
        return results

class ImageVisualizer:
    @staticmethod
    def plot_similar_images(query_image_path: str, 
                          similar_images: pd.DataFrame, 
                          reference_image_folder: str,
                          figsize=(15, 10)):
        """
        Plot the query image and its similar matches with annotations.
        
        Args:
            query_image_path: Path to the query image
            similar_images: DataFrame containing similar image results
            reference_image_folder: Path to the folder containing reference images
            figsize: Size of the figure (width, height)
        """
        n_similar = len(similar_images)
        n_cols = min(5, n_similar + 1)  # +1 for query image
        n_rows = math.ceil((n_similar + 1) / n_cols)
        
        # Create figure with gridspec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot query image
        ax = fig.add_subplot(gs[0, 0])
        query_img = Image.open(query_image_path).convert('RGB')
        ax.imshow(query_img)
        ax.set_title('Query Image\n' + os.path.basename(query_image_path), fontsize=8)
        ax.axis('off')
        
        # Plot similar images
        for idx, row in enumerate(similar_images.itertuples()):
            row_idx = (idx + 1) // n_cols
            col_idx = (idx + 1) % n_cols
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Load and display similar image
            img_path = os.path.join(reference_image_folder, row.filename) + ".png"
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                title = f'Match {idx+1}\nFile: {row.filename}\nSimilarity: {row.similarity:.3f}'
                ax.set_title(title, fontsize=8)
                ax.axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        return fig

class SingleImageSimilaritySearch:
    def __init__(self, checkpoint_path: str, reference_features, reference_filenames, 
                 train_df: pd.DataFrame):
        """
        Initialize the similarity search system.
        
        Args:
            checkpoint_path: Path to CLIP checkpoint
            reference_features: Features of reference images
            reference_filenames: Filenames of reference images
            train_df: DataFrame containing training data with categories
        """
        self.feature_extractor = CLIPEmbedder(checkpoint_path)
        self.faiss_index = FAISSIndex(reference_features, reference_filenames)
        self.faiss_index.build_index()
        
        # Prepare category mappings
        self.train_df = train_df
        
        # Create filename to category mappings
        self.train_df['id_as_filename'] = self.train_df['id'].astype(str).str.zfill(6) + '.png'
        self.train_categories = dict(zip(self.train_df['id_as_filename'], self.train_df['Category']))
        
        self.similarity_search = SimilaritySearch(
            self.faiss_index.index, 
            reference_filenames,
            self.train_categories,
            threshold=0
        )
        self.visualizer = ImageVisualizer()
    
    def get_image_category(self, image_filename: str) -> str:
        """Get category for an image filename."""
        return self.train_categories.get(image_filename, None)
    
    def find_similar_images(self, image_path: str, k: int = 10) -> pd.DataFrame:
        """Find similar images for a single query image within the same category."""
        # Get query image category
        query_filename = os.path.basename(image_path)
        query_category = self.get_image_category(query_filename)
        
        if query_category is None:
            raise ValueError(f"Category not found for image: {query_filename}")
        
        # Extract features and search
        query_features = self.feature_extractor.extract_single_feature(image_path)
        similar_images = self.similarity_search.search(query_features, query_category, k=k)
        
        results_df = pd.DataFrame(similar_images)
        results_df['query_image'] = query_filename
        results_df['query_category'] = query_category
        return results_df[['query_image', 'query_category', 'filename', 'similarity']]
    
    def find_and_visualize(self, 
                          image_path: str, 
                          reference_image_folder: str,
                          k: int = 10,
                          figsize=(25, 10)) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Find similar images within the same category and create visualization.
        
        Args:
            image_path: Path to the query image
            reference_image_folder: Path to the folder containing reference images
            k: Number of similar images to find
            figsize: Size of the output figure
            
        Returns:
            Tuple of (results DataFrame, matplotlib Figure)
        """
        results_df = self.find_similar_images(image_path, k)
        
        # Update visualizer to show category information
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(math.ceil((k + 1) / 5), min(5, k + 1), figure=fig)
        
        # Plot query image
        ax = fig.add_subplot(gs[0, 0])
        query_img = Image.open(image_path).convert('RGB')
        ax.imshow(query_img)
        query_category = results_df['query_category'].iloc[0]
        ax.set_title(f'Query Image\n{os.path.basename(image_path)}\nCategory: {query_category}', 
                    fontsize=8)
        ax.axis('off')
        
        # Plot similar images
        for idx, row in enumerate(results_df.itertuples()):
            row_idx = (idx + 1) // 5
            col_idx = (idx + 1) % 5
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img_path = os.path.join(reference_image_folder, row.filename) 
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                title = f'Match {idx+1}\nFile: {row.filename}\nSimilarity: {row.similarity:.3f}\nCategory: {row.query_category}'
                ax.set_title(title, fontsize=8)
                ax.axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        return results_df, fig
    

train_df = pd.read_csv('/scratch/data/m23csa016/meesho_data/new_train.csv')
# Load pre-computed embeddings
train_features_df = pd.read_parquet('/scratch/data/m23csa016/meesho_data/cvl_nobg_max_train_em_1.parquet')

feature_cols = [col for col in train_features_df.columns if col.startswith('feature_')]
train_features = train_features_df[feature_cols].values
train_filenames = train_features_df['filename'].tolist()

searcher = SingleImageSimilaritySearch(
    checkpoint_path="/scratch/data/m23csa016/meesho_data/bm_epoch_34_trainval_120024.pth",
    reference_features=train_features,
    reference_filenames=train_filenames,
    train_df=train_df
)


class CWTLabelCorrector:
    def __init__(self, 
                 similarity_searcher: SingleImageSimilaritySearch,
                 train_df: pd.DataFrame,
                 category_attribute_mapping: Dict[str, Dict[str, str]],
                 num_similar: int = 10,
                 similarity_threshold: float = 0.7,
                 log_dir: Optional[str] = None):
        """
        Initialize the category-aware training data label corrector with enhanced logging.
        
        Args:
            similarity_searcher: Initialized SingleImageSimilaritySearch object
            train_df: Training DataFrame containing the labels
            category_attribute_mapping: Mapping of categories to their attributes
            num_similar: Number of similar images to consider for majority voting
            similarity_threshold: Minimum similarity score to consider an image
            log_dir: Directory to save detailed logs (optional)
        """
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if not exists
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'label_correction_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create file handler for detailed logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'label_correction_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize other attributes
        self.searcher = similarity_searcher
        self.train_df = train_df
        self.category_attribute_mapping = category_attribute_mapping
        self.num_similar = num_similar
        self.similarity_threshold = similarity_threshold
        
        # Create a mapping from filename to row index for faster lookups
        self.train_df['filename'] = self.train_df['id'].astype(str).str.zfill(6) + '.png'
        self.filename_to_idx = dict(zip(self.train_df['filename'], self.train_df.index))
        
        # Create reverse mapping (attr_1 -> attribute name) for each category
        self.reverse_attribute_mapping = {
            category: {v: k for k, v in attrs.items()}
            for category, attrs in category_attribute_mapping.items()
        }
        
        # Log initialization details
        self.logger.info(f"Initialized CWTLabelCorrector")
        self.logger.info(f"Total training samples: {len(train_df)}")
        self.logger.info(f"Categories: {list(category_attribute_mapping.keys())}")
        self.logger.info(f"Similarity threshold: {similarity_threshold}")
        self.logger.info(f"Number of similar images to consider: {num_similar}")
        self.logger.info(f"Detailed logs will be saved to: {log_file}")

    def get_category_attributes(self, category: str) -> List[str]:
        """Get list of attribute columns for a given category."""
        if category not in self.category_attribute_mapping:
            return []
        return list(self.category_attribute_mapping[category].values())
        
    def get_attribute_name(self, category: str, attr_column: str) -> str:
        """Get the human-readable attribute name for a given category and column."""
        return self.reverse_attribute_mapping.get(category, {}).get(attr_column, attr_column)
    
        
    def _log_similar_images(self, image_path: str, similar_results: pd.DataFrame) -> None:
        """
        Log detailed information about similar images.
        
        Args:
            image_path: Path to the source image
            similar_results: DataFrame with similar images
        """
        self.logger.info(f"\n--- Similar Images for {os.path.basename(image_path)} ---")
        for _, row in similar_results.iterrows():
            self.logger.info(
                f"(Similarity: {row['similarity']:.4f})"
                f"Similar Image: /scratch/data/m23csa016/meesho_data/train_images_bg_removed/{row['filename']} "
            )
    
    def get_majority_label(self, similar_results: pd.DataFrame, 
                            category: str,
                            attr_column: str) -> Optional[Any]:
        """
        Find the majority label from similar images for a specific attribute.
        """
        # Previous implementation remains the same, but add logging
        # Filter by similarity threshold
        filtered_results = similar_results[
            similar_results['similarity'] >= self.similarity_threshold
        ]
        
        if len(filtered_results) == 0:
            self.logger.warning(
                f"No similar images above threshold for category {category}, "
                f"attribute {attr_column}"
            )
            return None
            
        # Get labels for similar images
        similar_labels = []
        for filename in filtered_results['filename']:
            if filename in self.filename_to_idx:
                idx = self.filename_to_idx[filename]
                label = self.train_df.at[idx, attr_column]
                if pd.notna(label):  # Only consider non-null labels
                    similar_labels.append(label)
        
        if not similar_labels:
            self.logger.warning(
                f"No valid labels found for similar images in "
                f"category {category}, attribute {attr_column}"
            )
            return None
            
        # Find majority label
        label_counts = Counter(similar_labels)
        majority_label, count = label_counts.most_common(1)[0]
        total_count = sum(label_counts.values())
        
        # Log label distribution
        self.logger.info(
            f"Label distribution for {category} - {attr_column}: "
            f"{dict(label_counts)} (Total: {total_count})"
        )
        
        # Return majority label only if it appears in more than 50% of similar images
        if count / total_count > 0.5:
            self.logger.info(
                f"Majority label found: {majority_label} "
                f"(Confidence: {count/total_count:.2%})"
            )
            return majority_label
        
        self.logger.info(
            f"No confident majority label for {category} - {attr_column}. "
            f"Keeping original label."
        )
        return None
    
    def correct_labels(self, image_folder: str) -> pd.DataFrame:
        """
        Correct labels in the training dataset using similarity-based majority voting.
        """
        corrected_df = self.train_df.copy()
        corrections_log = []
        total_processed = 0
        
        # Initialize correction counters for each category and attribute
        corrections_made = {
            category: {attr_name: 0 for attr_name in attrs.keys()}
            for category, attrs in self.category_attribute_mapping.items()
        }
        
        # Total images per category for statistical insights
        category_totals = self.train_df['Category'].value_counts().to_dict()
        
        for idx, row in self.train_df.iterrows():
            category = row['Category']
            if category not in self.category_attribute_mapping:
                continue
                
            image_path = f"{image_folder}/{str(row['id']).zfill(6)}.png"
            total_processed += 1
            
            try:
                # Find similar images
                similar_results = self.searcher.find_similar_images(
                    image_path, 
                    k=self.num_similar
                )
                
                # Log similar images details
                self._log_similar_images(image_path, similar_results)
                
                # Get relevant attributes for this category
                category_attrs = self.get_category_attributes(category)
                
                # Process each attribute
                for attr_column in category_attrs:
                    attr_name = self.get_attribute_name(category, attr_column)
                    
                    majority_label = self.get_majority_label(
                        similar_results,
                        category,
                        attr_column
                    )
                    
                    if majority_label is not None and (
                        pd.isna(row[attr_column]) or majority_label != row[attr_column]
                    ):
                        old_value = corrected_df.at[idx, attr_column]
                        corrected_df.at[idx, attr_column] = majority_label
                        corrections_made[category][attr_name] += 1
                        
                        # Log the correction with detailed information
                        corrections_log.append({
                            'image_id': row['id'],
                            'category': category,
                            'attribute': attr_name,
                            'attribute_column': attr_column,
                            'old_value': old_value,
                            'new_value': majority_label,
                            'num_similar_images': len(similar_results),
                            'avg_similarity': similar_results['similarity'].mean()
                        })
                        
                        self.logger.info(
                            f"Corrected label for Image {row['id']}:\n"
                            f"  Category: {category}\n"
                            f"  Attribute: {attr_name}\n"
                            f"  Old Value: {old_value}\n"
                            f"  New Value: {majority_label}"
                        )
                
                # Progress tracking
                if total_processed % 100 == 0:
                    self.logger.info(f"Processed {total_processed}/{len(self.train_df)} images")
                        
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {str(e)}")
                continue
                
        # Create corrections log DataFrame
        corrections_log_df = pd.DataFrame(corrections_log)
        
        # Log comprehensive correction statistics
        self.logger.warning("\nFinal Correction Statistics:")
        total_corrections = 0
        for category, attr_corrections in corrections_made.items():
            category_corrections = sum(attr_corrections.values())
            total_corrections += category_corrections
            category_total = category_totals.get(category, 0)
            
            self.logger.warning(f"\n{category}:")
            for attr_name, count in attr_corrections.items():
                percentage = (count / category_total) * 100 if category_total > 0 else 0
                self.logger.warning(f"  {attr_name}: {count} corrections ({percentage:.2f}%)")
            
            self.logger.warning(
                f"  Total corrections for {category}: {category_corrections} "
                f"({category_corrections/category_total*100:.2f}%)"
            )
        
        self.logger.warning(
            f"\nTotal Corrections Across All Categories: {total_corrections} "
            f"({total_corrections/len(self.train_df)*100:.2f}%)"
        )
        
        return corrected_df, corrections_log_df

def correct_training_labels(
    train_df: pd.DataFrame,
    image_folder: str,
    category_attribute_mapping: Dict[str, Dict[str, str]],
    output_path: str,
    log_dir: Optional[str] = None,
    num_similar: int = 10,
    similarity_threshold: float = 0.7
) -> None:
    """
    Correct labels in the training dataset and save the results with comprehensive logging.
    """
    # Configure global logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the label corrector
    corrector = CWTLabelCorrector(
        similarity_searcher=searcher,
        train_df=train_df,
        category_attribute_mapping=category_attribute_mapping,
        num_similar=num_similar,
        similarity_threshold=similarity_threshold,
        log_dir=log_dir
    )
    
    # Correct labels
    corrected_df, corrections_log_df = corrector.correct_labels(
        image_folder=image_folder
    )

    corrected_df.drop(columns=['id_as_filename', 'filename'], inplace=True)
    
    # Save corrected dataset
    corrected_df.to_csv(output_path, index=False)
    print(f"\nCorrected dataset saved to: {output_path}")
    
    # Save corrections log
    log_path = output_path.replace('.csv', '_corrections_log.csv')
    corrections_log_df.to_csv(log_path, index=False)
    print(f"Corrections log saved to: {log_path}")

# Run the label correction
correct_training_labels(
    train_df=train_df,
    image_folder='/scratch/data/m23csa016/meesho_data/train_images_bg_removed',
    category_attribute_mapping=category_class_attribute_mapping,  # Your mapping
    output_path='corrected_train_labels.csv',
    num_similar=30,
    similarity_threshold=0.75
)