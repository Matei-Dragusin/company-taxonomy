"""
Module for implementing semantic similarity using Sentence-BERT embeddings.
This approach provides better semantic understanding compared to TF-IDF or WordNet.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import logging
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SBERTProcessor:
    """
    Class for generating and managing Sentence-BERT embeddings
    for semantic similarity between companies and taxonomy.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', models_path: str = 'models/',
                use_gpu: bool = True, batch_size: int = 32):
        """
        Initialize the SBERT processor.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            models_path: Path to save model and embeddings
            use_gpu: Whether to use GPU for computation (if available)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.models_path = models_path
        self.batch_size = batch_size
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            logger.info(f"Created directory {models_path}")
        
        # Set device (GPU if available and requested, CPU otherwise)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load SBERT model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded SBERT model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading SBERT model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Generate SBERT embeddings for a list of texts.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress bar during encoding
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()
        
        # Ensure all inputs are strings
        processed_texts = [str(text) if text is not None else "" for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        logger.info(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
        return embeddings
    
    def transform_company_data(self, df: pd.DataFrame, text_column: str = 'combined_features') -> np.ndarray:
        """
        Transform company data into SBERT embeddings.
        
        Args:
            df: DataFrame with company data
            text_column: Column containing text to encode
            
        Returns:
            Array of company embeddings
        """
        logger.info(f"Transforming company data from column '{text_column}'")
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in company DataFrame")
            raise ValueError(f"Column '{text_column}' not found")
        
        # Extract text from DataFrame
        company_texts = df[text_column].fillna("").tolist()
        
        # Generate embeddings
        company_embeddings = self.generate_embeddings(company_texts)
        
        logger.info(f"Company embeddings shape: {company_embeddings.shape}")
        return company_embeddings
    
    def transform_taxonomy(self, taxonomy_df: pd.DataFrame, column: str = 'cleaned_label') -> np.ndarray:
        """
        Transform taxonomy labels into SBERT embeddings.
        
        Args:
            taxonomy_df: DataFrame with taxonomy data
            column: Column containing labels to encode
            
        Returns:
            Array of taxonomy embeddings
        """
        logger.info(f"Transforming taxonomy from column '{column}'")
        
        if column not in taxonomy_df.columns:
            logger.error(f"Column '{column}' not found in taxonomy DataFrame")
            raise ValueError(f"Column '{column}' not found")
        
        # Extract text from DataFrame
        taxonomy_texts = taxonomy_df[column].fillna("").tolist()
        
        # Generate embeddings
        taxonomy_embeddings = self.generate_embeddings(taxonomy_texts)
        
        logger.info(f"Taxonomy embeddings shape: {taxonomy_embeddings.shape}")
        return taxonomy_embeddings
    
    def compute_similarity(self, company_embeddings: np.ndarray, 
                         taxonomy_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between company and taxonomy embeddings.
        
        Args:
            company_embeddings: Embeddings of company data
            taxonomy_embeddings: Embeddings of taxonomy labels
            
        Returns:
            Similarity matrix
        """
        logger.info("Computing similarity between companies and taxonomy")
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(company_embeddings, taxonomy_embeddings)
        
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def get_top_matches(self, similarity_matrix: np.ndarray, 
                       taxonomy_df: pd.DataFrame,
                       top_k: int = 3, 
                       threshold: float = 0.1) -> List[List[Dict]]:
        """
        Get top taxonomy matches for each company based on similarity.
        
        Args:
            similarity_matrix: Matrix of similarity scores
            taxonomy_df: DataFrame with taxonomy data
            top_k: Maximum number of matches to return per company
            threshold: Minimum similarity threshold for a match
            
        Returns:
            List of lists with dictionary entries {label, score}
        """
        logger.info(f"Getting top {top_k} matches with threshold {threshold}")
        
        all_matches = []
        
        # For each company
        for i in range(similarity_matrix.shape[0]):
            company_similarities = similarity_matrix[i]
            
            # Get indices of top matches
            top_indices = company_similarities.argsort()[-top_k:][::-1]
            
            # Filter by threshold and create match dictionaries
            company_matches = []
            for idx in top_indices:
                score = company_similarities[idx]
                if score >= threshold:
                    company_matches.append({
                        'label': taxonomy_df.iloc[idx]['label'],
                        'score': float(score)
                    })
            
            all_matches.append(company_matches)
        
        return all_matches
    
    def assign_labels(self, companies_df: pd.DataFrame, 
                     taxonomy_df: pd.DataFrame,
                     top_k: int = 3, 
                     threshold: float = 0.1,
                     company_text_column: str = 'combined_features',
                     taxonomy_column: str = 'cleaned_label') -> pd.DataFrame:
        """
        Assign taxonomy labels to companies based on SBERT similarity.
        
        Args:
            companies_df: DataFrame with company data
            taxonomy_df: DataFrame with taxonomy data
            top_k: Maximum number of labels to assign
            threshold: Minimum similarity threshold
            company_text_column: Column in companies_df with text
            taxonomy_column: Column in taxonomy_df with labels
            
        Returns:
            DataFrame with assigned labels
        """
        logger.info(f"Assigning labels with top_k={top_k}, threshold={threshold}")
        
        # Transform data to embeddings
        company_embeddings = self.transform_company_data(companies_df, company_text_column)
        taxonomy_embeddings = self.transform_taxonomy(taxonomy_df, taxonomy_column)
        
        # Compute similarity
        similarity_matrix = self.compute_similarity(company_embeddings, taxonomy_embeddings)
        
        # Get top matches
        matches = self.get_top_matches(
            similarity_matrix, 
            taxonomy_df, 
            top_k=top_k, 
            threshold=threshold
        )
        
        # Add labels to DataFrame
        result_df = companies_df.copy()
        
        # Create lists of labels and scores
        result_df['insurance_labels'] = [
            [match['label'] for match in company_matches] 
            for company_matches in matches
        ]
        
        result_df['insurance_label_scores'] = [
            [match['score'] for match in company_matches] 
            for company_matches in matches
        ]
        
        # Create comma-separated string of labels
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        logger.info("Label assignment complete")
        
        return result_df
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings array to save
            filename: Name for the saved file
        """
        save_path = os.path.join(self.models_path, filename)
        joblib.dump(embeddings, save_path)
        logger.info(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, filename: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded embeddings array
        """
        load_path = os.path.join(self.models_path, filename)
        embeddings = joblib.load(load_path)
        logger.info(f"Loaded embeddings from {load_path}")
        return embeddings


# Example usage
if __name__ == "__main__":
    # Example usage of the SBERT processor
    from src.preprocessing.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file="companies.csv",
        taxonomy_file="insurance_taxonomy.csv"
    )
    
    # Initialize SBERT processor
    sbert_processor = SBERTProcessor(
        model_name='all-MiniLM-L6-v2',
        models_path='models/'
    )
    
    # Assign labels to companies
    classified_companies = sbert_processor.assign_labels(
        companies_df,
        taxonomy_df,
        top_k=3,
        threshold=0.1
    )
    
    # Save results
    classified_companies.to_csv('data/processed/sbert_classified_companies.csv', index=False)