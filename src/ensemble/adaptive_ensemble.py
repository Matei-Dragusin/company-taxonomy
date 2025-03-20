"""
Module for implementing an adaptive ensemble classifier that learns
optimal weights for combining different similarity approaches.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
import joblib
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import component processors
try:
    from src.feature_engineering.tfidf_processor import TFIDFProcessor
    from src.feature_engineering.sbert_processor import SBERTProcessor
except ImportError as e:
    logging.error(f"Error importing component processors: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveEnsembleClassifier:
    """
    An ensemble classifier that adaptively learns the optimal weights
    for combining different similarity measures.
    """
    
    def __init__(self, models_path: str = 'models/',
                initial_weights: Dict[str, float] = None,
                weight_learning_rate: float = 0.01,
                use_sbert: bool = True,
                sbert_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the adaptive ensemble classifier.
        
        Args:
            models_path: Path to save/load models
            initial_weights: Initial weights for different similarity methods
            weight_learning_rate: Learning rate for weight adaptation
            use_sbert: Whether to use Sentence-BERT embeddings
            sbert_model_name: SBERT model to use
        """
        self.models_path = models_path
        self.weight_learning_rate = weight_learning_rate
        self.use_sbert = use_sbert
        
        # Set default weights if not provided
        if initial_weights is None:
            self.weights = {
                'tfidf': 0.4,
                'sbert': 0.4,
                'keyword': 0.2
            }
        else:
            self.weights = initial_weights
        
        # Ensure weights sum to 1
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for k in self.weights:
                self.weights[k] /= weight_sum
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            logger.info(f"Created directory {models_path}")
        
        # Initialize component processors
        self.tfidf_processor = TFIDFProcessor(
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 3),
            models_path=models_path
        )
        
        if use_sbert:
            self.sbert_processor = SBERTProcessor(
                model_name=sbert_model_name,
                models_path=models_path
            )
        
        # Initialize meta-model for weight learning
        self.meta_model = None
        self.scaler = StandardScaler()
        
        # Storage for similarity matrices
        self.similarity_matrices = {}
        
    def generate_similarity_matrices(self, companies_df: pd.DataFrame, 
                                   taxonomy_df: pd.DataFrame,
                                   company_text_column: str = 'combined_features') -> Dict[str, np.ndarray]:
        """
        Generate similarity matrices from all component methods.
        
        Args:
            companies_df: DataFrame with company data
            taxonomy_df: DataFrame with taxonomy data
            company_text_column: Column with company text
            
        Returns:
            Dictionary of similarity matrices keyed by method name
        """
        logger.info("Generating similarity matrices from all component methods")
        matrices = {}
        
        # Generate TF-IDF similarity matrix
        try:
            logger.info("Generating TF-IDF similarity")
            # Fit vectorizer
            self.tfidf_processor.fit_vectorizer(
                companies_df, 
                taxonomy_df, 
                company_text_column=company_text_column,
                taxonomy_column='cleaned_label'
            )
            
            # Generate TF-IDF vectors
            company_tfidf = self.tfidf_processor.transform_company_data(
                companies_df, 
                text_column=company_text_column
            )
            
            taxonomy_tfidf = self.tfidf_processor.transform_taxonomy(
                taxonomy_df, 
                column='cleaned_label'
            )
            
            # Compute TF-IDF similarity
            tfidf_similarity = cosine_similarity(company_tfidf, taxonomy_tfidf)
            matrices['tfidf'] = tfidf_similarity
            logger.info(f"TF-IDF similarity matrix shape: {tfidf_similarity.shape}")
        except Exception as e:
            logger.error(f"Error generating TF-IDF similarity: {e}")
            matrices['tfidf'] = None
        
        # Generate SBERT similarity matrix if enabled
        if self.use_sbert:
            try:
                logger.info("Generating SBERT similarity")
                # Generate SBERT embeddings
                company_embeddings = self.sbert_processor.transform_company_data(
                    companies_df, 
                    text_column=company_text_column
                )
                
                taxonomy_embeddings = self.sbert_processor.transform_taxonomy(
                    taxonomy_df, 
                    column='cleaned_label'
                )
                
                # Compute SBERT similarity
                sbert_similarity = cosine_similarity(company_embeddings, taxonomy_embeddings)
                matrices['sbert'] = sbert_similarity
                logger.info(f"SBERT similarity matrix shape: {sbert_similarity.shape}")
                
                # Save embeddings for future use
                self.sbert_processor.save_embeddings(
                    company_embeddings, 
                    "company_sbert_embeddings.joblib"
                )
                
                self.sbert_processor.save_embeddings(
                    taxonomy_embeddings, 
                    "taxonomy_sbert_embeddings.joblib"
                )
            except Exception as e:
                logger.error(f"Error generating SBERT similarity: {e}")
                matrices['sbert'] = None
        
        # Generate keyword-based similarity matrix
        try:
            logger.info("Generating keyword-based similarity")
            
            # Create a matrix to store keyword similarity
            keyword_similarity = np.zeros((len(companies_df), len(taxonomy_df)))
            
            # For each company, compute keyword similarity with each taxonomy label
            from src.ensemble.ensemble_classifier import EnsembleClassifier
            temp_classifier = EnsembleClassifier(models_path=self.models_path)
            
            # Preprocess taxonomy for keyword matching
            enhanced_taxonomy_df = temp_classifier.preprocess_taxonomy(taxonomy_df)
            
            # For each company
            for i, (_, company) in enumerate(companies_df.iterrows()):
                company_text = company[company_text_column] if company_text_column in company else ""
                
                # Compute keyword similarity
                keyword_sim = temp_classifier.compute_keyword_similarity(
                    company_text, 
                    enhanced_taxonomy_df
                )
                
                keyword_similarity[i] = keyword_sim
            
            matrices['keyword'] = keyword_similarity
            logger.info(f"Keyword similarity matrix shape: {keyword_similarity.shape}")
        except Exception as e:
            logger.error(f"Error generating keyword similarity: {e}")
            matrices['keyword'] = None
        
        # Store all matrices
        self.similarity_matrices = matrices
        
        return matrices
    
    def train_meta_model(self, companies_df: pd.DataFrame, 
                        taxonomy_df: pd.DataFrame,
                        company_text_column: str = 'combined_features',
                        sample_size: int = 100,
                        test_size: float = 0.2) -> None:
        """
        Train a meta-model to learn optimal weights for the ensemble.
        
        Args:
            companies_df: DataFrame with company data
            taxonomy_df: DataFrame with taxonomy data
            company_text_column: Column with company text
            sample_size: Number of samples to use for training
            test_size: Portion of data to use for testing
        """
        logger.info(f"Training meta-model with {sample_size} samples")
        
        # Generate similarity matrices if not already done
        if not self.similarity_matrices:
            self.generate_similarity_matrices(
                companies_df,
                taxonomy_df,
                company_text_column
            )
        
        # Extract required matrices
        matrices = []
        for method in ['tfidf', 'sbert', 'keyword']:
            if method in self.similarity_matrices and self.similarity_matrices[method] is not None:
                matrices.append(self.similarity_matrices[method])
        
        if not matrices:
            logger.error("No valid similarity matrices found for meta-model training")
            return
        
        # Sample company and taxonomy indices
        num_companies = matrices[0].shape[0]
        num_taxonomy = matrices[0].shape[1]
        
        # Limit sample size to available data
        sample_size = min(sample_size, num_companies * num_taxonomy)
        
        # Create training data
        X = []
        y = []
        
        # Sample random pairs of companies and taxonomy labels
        for _ in range(sample_size):
            company_idx = np.random.randint(0, num_companies)
            taxonomy_idx = np.random.randint(0, num_taxonomy)
            
            # Extract similarity scores from each method
            features = []
            for matrix in matrices:
                features.append(matrix[company_idx, taxonomy_idx])
            
            X.append(features)
            
            # For training, assume high similarity scores in any method indicate relevance
            # This is a heuristic approach since we don't have labeled data
            max_similarity = max(features)
            y.append(1 if max_similarity > 0.1 else 0)  # Use 0.1 as threshold for relevance
        
        # Split data
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train logistic regression as meta-model
        self.meta_model = LogisticRegression(class_weight='balanced')
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate on test set
        train_accuracy = self.meta_model.score(X_train, y_train)
        test_accuracy = self.meta_model.score(X_test, y_test)
        
        logger.info(f"Meta-model training complete:")
        logger.info(f"  - Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  - Test accuracy: {test_accuracy:.4f}")
        
        # Update weights based on meta-model coefficients
        if hasattr(self.meta_model, 'coef_'):
            coeffs = self.meta_model.coef_[0]
            methods = []
            
            if 'tfidf' in self.similarity_matrices and self.similarity_matrices['tfidf'] is not None:
                methods.append('tfidf')
            
            if 'sbert' in self.similarity_matrices and self.similarity_matrices['sbert'] is not None:
                methods.append('sbert')
                
            if 'keyword' in self.similarity_matrices and self.similarity_matrices['keyword'] is not None:
                methods.append('keyword')
            
            # Ensure coefficients are positive
            coeffs = np.abs(coeffs)
            
            # Normalize coefficients to sum to 1
            if np.sum(coeffs) > 0:
                coeffs = coeffs / np.sum(coeffs)
                
                # Update weights
                for i, method in enumerate(methods):
                    self.weights[method] = float(coeffs[i])
                
                logger.info(f"Updated weights: {self.weights}")
            else:
                logger.warning("Could not update weights: all coefficients are zero")
    
    def ensemble_classify(self, companies_df: pd.DataFrame, 
                         taxonomy_df: pd.DataFrame,
                         top_k: int = 3, 
                         threshold: float = 0.08,
                         company_text_column: str = 'combined_features',
                         train_meta_model: bool = True,
                         meta_model_samples: int = 100) -> pd.DataFrame:
        """
        Classify companies using the adaptive ensemble approach.
        
        Args:
            companies_df: DataFrame with company data
            taxonomy_df: DataFrame with taxonomy data
            top_k: Maximum number of labels to assign
            threshold: Minimum similarity threshold
            company_text_column: Column with company text
            train_meta_model: Whether to train the meta-model
            meta_model_samples: Number of samples for meta-model training
            
        Returns:
            DataFrame with classification results
        """
        logger.info(f"Starting adaptive ensemble classification with top_k={top_k}, threshold={threshold}")
        start_time = time.time()
        
        # Generate similarity matrices
        self.generate_similarity_matrices(
            companies_df,
            taxonomy_df,
            company_text_column
        )
        
        # Train meta-model if requested
        if train_meta_model:
            self.train_meta_model(
                companies_df,
                taxonomy_df,
                company_text_column,
                sample_size=meta_model_samples
            )
        
        # Create result DataFrame
        result_df = companies_df.copy()
        all_matches = []
        all_scores = []
        
        # For each company, compute weighted similarity and get top matches
        logger.info("Computing ensemble similarity and assigning labels")
        
        # Get total number of companies and taxonomy labels
        num_companies = len(companies_df)
        num_taxonomy = len(taxonomy_df)
        
        # Initialize ensemble similarity matrix
        ensemble_similarity = np.zeros((num_companies, num_taxonomy))
        
        # Combine similarity matrices with learned weights
        for method, weight in self.weights.items():
            if method in self.similarity_matrices and self.similarity_matrices[method] is not None:
                ensemble_similarity += weight * self.similarity_matrices[method]
        
        # Get top matches for each company
        for i in range(num_companies):
            company_similarity = ensemble_similarity[i]
            
            # Get indices of top matches
            top_indices = company_similarity.argsort()[-top_k:][::-1]
            
            # Filter by threshold and create match dictionaries
            company_matches = []
            company_scores = []
            
            for idx in top_indices:
                score = company_similarity[idx]
                if score >= threshold:
                    company_matches.append(taxonomy_df.iloc[idx]['label'])
                    company_scores.append(float(score))
            
            all_matches.append(company_matches)
            all_scores.append(company_scores)
        
        # Add matches to result DataFrame
        result_df['insurance_labels'] = all_matches
        result_df['insurance_label_scores'] = all_scores
        
        # Create comma-separated string of labels
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        logger.info(f"Classification complete in {time.time() - start_time:.2f} seconds")
        logger.info(f"Final weights used: {self.weights}")
        
        return result_df
    
    def save_model(self, filename: str = 'adaptive_ensemble_model.joblib') -> None:
        """
        Save the adaptive ensemble model.
        
        Args:
            filename: Name for the saved model file
        """
        save_path = os.path.join(self.models_path, filename)
        
        model_data = {
            'weights': self.weights,
            'meta_model': self.meta_model,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Saved adaptive ensemble model to {save_path}")
    
    def load_model(self, filename: str = 'adaptive_ensemble_model.joblib') -> bool:
        """
        Load a saved adaptive ensemble model.
        
        Args:
            filename: Name of the model file to load
            
        Returns:
            Boolean indicating success
        """
        load_path = os.path.join(self.models_path, filename)
        
        try:
            model_data = joblib.load(load_path)
            
            self.weights = model_data['weights']
            self.meta_model = model_data['meta_model']
            self.scaler = model_data['scaler']
            
            logger.info(f"Loaded adaptive ensemble model from {load_path}")
            logger.info(f"Loaded weights: {self.weights}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading adaptive ensemble model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Example usage of the adaptive ensemble classifier
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
    
    # Initialize adaptive ensemble classifier
    classifier = AdaptiveEnsembleClassifier(
        models_path='models/',
        use_sbert=True
    )
    
    # Classify companies
    classified_companies = classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=3,
        threshold=0.08,
        train_meta_model=True
    )
    
    # Save results and model
    classified_companies.to_csv('data/processed/adaptive_ensemble_classified_companies.csv', index=False)
    classifier.save_model()