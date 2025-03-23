"""
Module for implementing TF-IDF and feature engineering
for classifying companies into insurance taxonomy.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import joblib
import os

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TFIDFProcessor:
    """
    Class for generating and managing TF-IDF representations
    for company texts and insurance taxonomy.
    """
    
    def __init__(self, min_df: int = 1, max_df: float = 0.9, 
                 ngram_range: Tuple[int, int] = (1, 3), models_path: str = 'models/'):
        """
        Initializes the TF-IDF processor.
        
        Args:
            min_df: Minimum document frequency to include a term
            max_df: Maximum document frequency to include a term (removes overly common words)
            ngram_range: Range of n-grams for vectorization (1,2 = unigrams and bigrams)
            models_path: Path where trained models will be saved
        """
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.models_path = models_path
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            logger.info(f"Created directory {models_path}")
        
        # Initialize a single TF-IDF vectorizer for both companies and taxonomy
        # This ensures all vectors are in the same vector space
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True, # Apply logarithmic scaling to term frequency
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
    
    def fit_vectorizer(self, companies_df: pd.DataFrame, taxonomy_df: pd.DataFrame, 
                      company_text_column: str = 'combined_features',
                      taxonomy_column: str = 'cleaned_label') -> None:
        """
        Trains the TF-IDF vectorizer on combined company and taxonomy data.
        This ensures all vectors will be in the same vector space.
        
        Args:
            companies_df: DataFrame with preprocessed company data
            taxonomy_df: DataFrame with preprocessed insurance taxonomy
            company_text_column: Column in companies_df containing text for vectorization
            taxonomy_column: Column in taxonomy_df containing text for vectorization
        """
        logger.info("Training TF-IDF vectorizer on combined data (companies + taxonomy)")
        
        # Check for required columns
        if company_text_column not in companies_df.columns:
            logger.error(f"Column {company_text_column} does not exist in the company dataset!")
            return
            
        if taxonomy_column not in taxonomy_df.columns:
            logger.error(f"Column {taxonomy_column} does not exist in the taxonomy dataset!")
            return
        
        # Combine texts from companies and taxonomy for training the vectorizer
        company_texts = companies_df[company_text_column].fillna('').tolist()
        taxonomy_texts = taxonomy_df[taxonomy_column].fillna('').tolist()
        
        # Combine all texts and train the vectorizer on them
        all_texts = company_texts + taxonomy_texts
        logger.info(f"Training vectorizer on {len(all_texts)} combined texts")
        
        self.vectorizer.fit(all_texts)
        
        # Save the vocabulary size
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vocabulary created with {self.vocabulary_size} terms")
    
    def transform_text(self, texts: List[str]) -> np.ndarray:
        """
        Transforms texts into TF-IDF representations.
        
        Args:
            texts: List of texts to be transformed
            
        Returns:
            TF-IDF matrix for the given texts
        """
        if not hasattr(self, 'vectorizer') or not hasattr(self.vectorizer, 'vocabulary_'):
            logger.error("The vectorizer has not been trained! Call fit_vectorizer() first.")
            return None
            
        return self.vectorizer.transform(texts)
    
    def transform_company_data(self, df: pd.DataFrame, text_column: str = 'combined_features') -> np.ndarray:
        """
        Transforms company data into TF-IDF representations.
        
        Args:
            df: DataFrame with preprocessed company data
            text_column: Column containing text for transformation
            
        Returns:
            TF-IDF matrix for the given companies
        """
        logger.info(f"Transforming company data from column {text_column} into TF-IDF representations")
        
        if text_column not in df.columns:
            logger.error(f"Column {text_column} does not exist in the company dataset!")
            return None
        
        company_texts = df[text_column].fillna('').tolist()
        company_vectors = self.transform_text(company_texts)
        
        logger.info(f"Transformation complete - matrix shape: {company_vectors.shape}")
        return company_vectors
    
    def transform_taxonomy(self, taxonomy_df: pd.DataFrame, column: str = 'cleaned_label') -> np.ndarray:
        """
        Transforms taxonomy labels into TF-IDF representations.
        
        Args:
            taxonomy_df: DataFrame with preprocessed insurance taxonomy
            column: Column containing cleaned labels
            
        Returns:
            TF-IDF matrix for the taxonomy labels
        """
        logger.info(f"Transforming taxonomy labels from column {column}")
        
        if column not in taxonomy_df.columns:
            logger.error(f"Column {column} does not exist in the taxonomy dataset!")
            return None
        
        taxonomy_texts = taxonomy_df[column].fillna('').tolist()
        taxonomy_vectors = self.transform_text(taxonomy_texts)
        
        logger.info(f"Transformation complete - matrix shape: {taxonomy_vectors.shape}")
        return taxonomy_vectors
    
    def compute_similarity_to_taxonomy(self, company_vectors: np.ndarray, 
                                      taxonomy_vectors: np.ndarray) -> np.ndarray:
        """
        Computes similarity between company vectors and taxonomy labels.
        
        Args:
            company_vectors: TF-IDF representations of companies
            taxonomy_vectors: TF-IDF representations of taxonomy labels
            
        Returns:
            Similarity matrix between companies and labels
        """
        logger.info("Computing similarity between companies and labels")
        
        # Compute cosine similarity between companies and labels
        similarity_matrix = cosine_similarity(company_vectors, taxonomy_vectors)
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def get_top_taxonomy_matches(self, similarity_matrix: np.ndarray, 
                               taxonomy_df: pd.DataFrame,
                               top_k: int = 3, 
                               threshold: float = 0.1) -> List[List[Dict]]:
        """
        Gets the top matches from taxonomy for each company.
        
        Args:
            similarity_matrix: Similarity matrix between companies and labels
            taxonomy_df: DataFrame with taxonomy
            top_k: Maximum number of labels to return for each company
            threshold: Minimum similarity threshold to consider a match
            
        Returns:
            List of lists with dictionaries containing matched labels and scores
        """
        logger.info(f"Getting top {top_k} matches for each company (threshold: {threshold})")
        
        all_matches = []
        
        # For each company
        for i in range(similarity_matrix.shape[0]):
            company_similarities = similarity_matrix[i]
            
            # Get sorted indices of the most similar labels
            top_indices = company_similarities.argsort()[-top_k:][::-1]
            
            # Filter based on the threshold
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
    
    def assign_insurance_labels(self, df: pd.DataFrame, 
                               taxonomy_df: pd.DataFrame,
                               top_k: int = 3, 
                               threshold: float = 0.1,
                               company_text_column: str = 'combined_features',
                               taxonomy_column: str = 'cleaned_label') -> pd.DataFrame:
        """
        Assigns insurance labels to companies based on TF-IDF similarity.
        
        Args:
            df: DataFrame with preprocessed company data
            taxonomy_df: DataFrame with insurance taxonomy
            top_k: Maximum number of labels to assign to a company
            threshold: Minimum similarity threshold to assign a label
            company_text_column: Column in df containing text for comparison
            taxonomy_column: Column in taxonomy_df containing text for comparison
            
        Returns:
            Original DataFrame with a new column 'insurance_label' added
        """
        logger.info(f"Assigning insurance labels for {len(df)} companies")
        
        # Train the vectorizer on combined data if not already trained
        if not hasattr(self, 'vocabulary_size'):
            self.fit_vectorizer(df, taxonomy_df, company_text_column, taxonomy_column)
        
        # Transform company and taxonomy data into TF-IDF vectors
        company_vectors = self.transform_company_data(df, company_text_column)
        taxonomy_vectors = self.transform_taxonomy(taxonomy_df, taxonomy_column)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_to_taxonomy(company_vectors, taxonomy_vectors)
        
        # Get the top matches for each company
        matches = self.get_top_taxonomy_matches(
            similarity_matrix, 
            taxonomy_df, 
            top_k=top_k, 
            threshold=threshold
        )
        
        # Add labels to the DataFrame
        result_df = df.copy()
        
        # Create a column with lists of labels and a column with lists of scores
        result_df['insurance_labels'] = [
            [match['label'] for match in company_matches] 
            for company_matches in matches
        ]
        
        result_df['insurance_label_scores'] = [
            [match['score'] for match in company_matches] 
            for company_matches in matches
        ]
        
        # Create a 'insurance_label' column with labels separated by commas
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        logger.info("Label assignment complete")
        
        return result_df
    
    def save_models(self, filename_prefix: str = 'tfidf_model') -> None:
        """
        Saves the trained TF-IDF model.
        
        Args:
            filename_prefix: Prefix for the saved file name
        """
        logger.info(f"Saving TF-IDF model to directory {self.models_path}")
        
        if hasattr(self, 'vectorizer') and hasattr(self.vectorizer, 'vocabulary_'):
            save_path = os.path.join(self.models_path, f"{filename_prefix}.joblib")
            joblib.dump(self.vectorizer, save_path)
            logger.info(f"Model saved at {save_path}")
        else:
            logger.warning("No trained model found to save")
    
    def load_models(self, filepath: str) -> bool:
        """
        Loads a pre-trained TF-IDF model.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Bool indicating whether the loading was successful
        """
        logger.info(f"Loading TF-IDF model from {filepath}")
        
        try:
            self.vectorizer = joblib.load(filepath)
            
            if hasattr(self.vectorizer, 'vocabulary_'):
                self.vocabulary_size = len(self.vectorizer.vocabulary_)
                logger.info(f"Model loaded successfully: vocabulary with {self.vocabulary_size} terms")
                return True
            else:
                logger.error("The loaded model does not have a valid vocabulary")
                return False
        except Exception as e:
            logger.error(f"Error loading the model: {e}")
            return False

if __name__ == "__main__":
    
    from src.preprocessing.preprocessing import DataPreprocessor
    
    # Initialize the data preprocessor
    data_preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Load and preprocess the data
    companies_df, taxonomy_df = data_preprocessor.process_company_and_taxonomy(
        company_file="companies.csv",
        taxonomy_file="insurance_taxonomy.csv"
    )
    
    # Initialize the TF-IDF processor
    tfidf_processor = TFIDFProcessor(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 3),
        models_path='models/'
    )
    
    # Assign insurance labels to companies
    # (fit_vectorizer will be automatically called in assign_insurance_labels)
    classified_companies = tfidf_processor.assign_insurance_labels(
        companies_df,
        taxonomy_df,
        top_k=3,
        threshold=0.05,
        company_text_column='combined_features',
        taxonomy_column='cleaned_label'
    )
    
    # Save the results
    classified_companies.to_csv('data/processed/classified_companies.csv', index=False)
    
    # Save the models for future use
    tfidf_processor.save_models()
