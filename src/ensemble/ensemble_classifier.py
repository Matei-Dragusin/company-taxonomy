"""
Module for implementing an Ensemble Classifier for insurance taxonomy,
combining TF-IDF with other approaches.
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
import re
import nltk
from nltk.corpus import wordnet
from src.feature_engineering.tfidf_processor import TFIDFProcessor

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleClassifier:
    """
    Ensemble Classifier that combines multiple approaches to
    classify companies into the insurance taxonomy.
    """
    
    def __init__(self, models_path: str = 'models/', 
                 wordnet_weight: float = 0.3,
                 keyword_weight: float = 0.2,
                 tfidf_weight: float = 0.5):
        """
        Initialize the Ensemble classifier.
        
        Args:
            models_path: Path for saving/loading models
            wordnet_weight: Weight for the WordNet-based approach
            keyword_weight: Weight for the keyword-based approach
            tfidf_weight: Weight for the TF-IDF approach
        """
        self.models_path = models_path
        
        # Weights for combining different approaches
        self.weights = {
            'tfidf': tfidf_weight,
            'wordnet': wordnet_weight,
            'keyword': keyword_weight
        }
        
        # Initialize the TF-IDF processor
        self.tfidf_processor = TFIDFProcessor(
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 3),
            models_path=models_path
        )
        
        # Relevant keywords for each insurance industry sector
        self.insurance_keywords = {
            'property': ['building', 'real estate', 'property', 'housing', 'commercial property', 
                        'residential', 'construction', 'office', 'earthquake', 'flood'],
            'liability': ['professional', 'malpractice', 'errors', 'omissions', 'directors', 
                          'officers', 'product liability', 'public liability', 'employer liability'],
            'health': ['medical', 'health', 'dental', 'vision', 'disability', 'wellness', 
                       'illness', 'hospital', 'clinical', 'pharmaceutical'],
            'life': ['life', 'death', 'terminal', 'mortality', 'retirement', 'pension', 
                     'annuity', 'saving', 'investment'],
            'auto': ['vehicle', 'car', 'truck', 'fleet', 'commercial auto', 'personal auto', 
                     'driver', 'collision', 'comprehensive', 'transportation'],
            'cyber': ['data', 'breach', 'privacy', 'security', 'cyber', 'information', 
                      'digital', 'hack', 'technology', 'internet'],
            'business': ['business', 'commercial', 'enterprise', 'corporate', 'company', 
                         'interruption', 'operation', 'revenue', 'income'],
            'workers_comp': ['employee', 'worker', 'injury', 'compensation', 'occupational', 
                             'workplace', 'safety', 'accident', 'disability', 'hazard'],
            'specialty': ['marine', 'aviation', 'travel', 'event', 'pet', 'art', 'agriculture', 
                          'energy', 'environmental', 'political']
        }
        
        # Ensure we have NLTK WordNet available
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download WordNet: {e}")
    
    def preprocess_taxonomy(self, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess taxonomy data for classification.
        
        Args:
            taxonomy_df: DataFrame with insurance taxonomy
            
        Returns:
            DataFrame with preprocessed taxonomy
        """
        logger.info("Preprocessing taxonomy for the Ensemble classifier")
        
        df_result = taxonomy_df.copy()
        
        # Extract key terms from each label
        if 'cleaned_label' in df_result.columns:
            df_result['key_terms'] = df_result['cleaned_label'].apply(self._extract_key_terms)
            
            # Generate synonyms for key terms using WordNet
            df_result['synonyms'] = df_result['key_terms'].apply(self._generate_synonyms)
            
            # Map labels to relevant insurance sectors
            df_result['insurance_sectors'] = df_result['cleaned_label'].apply(
                lambda x: self._map_to_insurance_sectors(x)
            )
        
        logger.info("Taxonomy preprocessing complete")
        
        return df_result
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key terms
        """
        if not isinstance(text, str):
            return []
            
        # Split text into words and remove too short words
        words = text.lower().split()
        key_terms = [word for word in words if len(word) > 3]
        
        # Add n-grams (phrases of 2-3 words)
        for n in range(2, 4):
            for i in range(len(words) - n + 1):
                key_terms.append(' '.join(words[i:i+n]))
        
        return key_terms
    
    def _generate_synonyms(self, terms: List[str]) -> List[str]:
        """
        Generate synonyms for given terms using WordNet.
        
        Args:
            terms: List of terms to generate synonyms for
            
        Returns:
            Extended list with original terms and their synonyms
        """
        all_terms = list(terms)  # Start with original terms
        
        for term in terms:
            # Skip phrases, use only single words
            if ' ' in term:
                continue
                
            # Look for synonyms in WordNet
            for synset in wordnet.synsets(term):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and synonym not in all_terms:
                        all_terms.append(synonym)
        
        return all_terms
    
    def _map_to_insurance_sectors(self, text: str) -> List[str]:
        """
        Map a text to relevant insurance sectors based on keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relevant insurance sectors
        """
        if not isinstance(text, str):
            return []
            
        relevant_sectors = []
        text_lower = text.lower()
        
        for sector, keywords in self.insurance_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    relevant_sectors.append(sector)
                    break
        
        return relevant_sectors
    
    def compute_keyword_similarity(self, company_text: str, 
                                   taxonomy_df: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity based on insurance keyword matching.
        
        Args:
            company_text: Preprocessed company text
            taxonomy_df: Preprocessed taxonomy DataFrame
            
        Returns:
            Array of similarity scores for each taxonomy label
        """
        if not isinstance(company_text, str):
            company_text = ""
            
        company_text = company_text.lower()
        company_sectors = self._map_to_insurance_sectors(company_text)
        
        similarities = []
        
        for _, row in taxonomy_df.iterrows():
            # Get sectors for this taxonomy label
            label_sectors = row.get('insurance_sectors', [])
            
            # Calculate Jaccard similarity between company and label sectors
            if not company_sectors or not label_sectors:
                similarities.append(0.0)
            else:
                intersection = len(set(company_sectors) & set(label_sectors))
                union = len(set(company_sectors) | set(label_sectors))
                similarities.append(intersection / union if union > 0 else 0.0)
        
        return np.array(similarities)
    
    def compute_wordnet_similarity(self, company_text: str,
                                  taxonomy_df: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity based on WordNet synonyms matching.
        
        Args:
            company_text: Preprocessed company text
            taxonomy_df: Preprocessed taxonomy DataFrame
            
        Returns:
            Array of similarity scores for each taxonomy label
        """
        if not isinstance(company_text, str):
            company_text = ""
            
        # Extract key terms from company text
        company_terms = self._extract_key_terms(company_text)
        # Generate synonyms for company terms
        company_synonyms = self._generate_synonyms(company_terms)
        
        similarities = []
        
        for _, row in taxonomy_df.iterrows():
            # Get synonyms for this taxonomy label
            label_synonyms = row.get('synonyms', [])
            
            # Calculate similarity based on term overlap
            if not company_synonyms or not label_synonyms:
                similarities.append(0.0)
            else:
                common_terms = set(company_synonyms) & set(label_synonyms)
                similarity = len(common_terms) / (len(company_synonyms) + len(label_synonyms) - len(common_terms))
                similarities.append(similarity)
        
        return np.array(similarities)
    
    def ensemble_classify(self, companies_df: pd.DataFrame, 
                         taxonomy_df: pd.DataFrame,
                         top_k: int = 5, 
                         threshold: float = 0.08,
                         company_text_column: str = 'combined_features') -> pd.DataFrame:
        """
        Classify companies using the ensemble approach.
        
        Args:
            companies_df: DataFrame with preprocessed company data
            taxonomy_df: DataFrame with preprocessed taxonomy data
            top_k: Maximum number of labels to assign to a company
            threshold: Minimum similarity threshold to assign a label
            company_text_column: Column containing text for comparison
            
        Returns:
            DataFrame with added classification results
        """
        logger.info(f"Starting ensemble classification for {len(companies_df)} companies")
        
        # Preprocess taxonomy for ensemble methods
        enhanced_taxonomy_df = self.preprocess_taxonomy(taxonomy_df)
        
        # First get TF-IDF classification
        tfidf_classified = self.tfidf_processor.assign_insurance_labels(
            companies_df,
            taxonomy_df,
            top_k=top_k*2,  # Get more candidates than we need
            threshold=threshold/2,  # Lower threshold to get more candidates
            company_text_column=company_text_column,
            taxonomy_column='cleaned_label'
        )
        
        # Extract the similarity matrix from TF-IDF
        tfidf_vectors = self.tfidf_processor.transform_company_data(companies_df, company_text_column)
        taxonomy_vectors = self.tfidf_processor.transform_taxonomy(taxonomy_df, 'cleaned_label')
        tfidf_similarities = cosine_similarity(tfidf_vectors, taxonomy_vectors)
        
        # Create the final results DataFrame
        result_df = companies_df.copy()
        all_ensemble_matches = []
        all_ensemble_scores = []
        
        # Process each company
        for i, (_, company) in enumerate(companies_df.iterrows()):
            company_text = company[company_text_column] if company_text_column in company.index else ""
            
            # Get TF-IDF similarity for this company
            company_tfidf_sim = tfidf_similarities[i]
            
            # Get keyword-based similarity
            company_keyword_sim = self.compute_keyword_similarity(company_text, enhanced_taxonomy_df)
            
            # Get WordNet-based similarity
            company_wordnet_sim = self.compute_wordnet_similarity(company_text, enhanced_taxonomy_df)
            
            # Combine similarities using weighted average
            ensemble_sim = (
                self.weights['tfidf'] * company_tfidf_sim +
                self.weights['keyword'] * company_keyword_sim +
                self.weights['wordnet'] * company_wordnet_sim
            )
            
            # Get top matches
            top_indices = ensemble_sim.argsort()[-top_k:][::-1]
            
            # Filter by threshold
            company_matches = []
            company_scores = []
            
            for idx in top_indices:
                score = ensemble_sim[idx]
                if score >= threshold:
                    company_matches.append(enhanced_taxonomy_df.iloc[idx]['label'])
                    company_scores.append(float(score))
            
            all_ensemble_matches.append(company_matches)
            all_ensemble_scores.append(company_scores)
        
        # Add results to DataFrame
        result_df['insurance_labels'] = all_ensemble_matches
        result_df['insurance_label_scores'] = all_ensemble_scores
        
        # Create comma-separated string of labels
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        logger.info("Ensemble classification complete")
        
        return result_df
    
    def save_models(self, filename_prefix: str = 'ensemble_model') -> None:
        """
        Save the ensemble model and its components.
        
        Args:
            filename_prefix: Prefix for the filename when saving
        """
        logger.info(f"Saving ensemble model to {self.models_path}")
        
        # Save TF-IDF processor
        self.tfidf_processor.save_models(filename_prefix='tfidf_component')
        
        # Save ensemble configuration
        ensemble_config = {
            'weights': self.weights,
            'insurance_keywords': self.insurance_keywords
        }
        
        config_path = os.path.join(self.models_path, f"{filename_prefix}_config.joblib")
        joblib.dump(ensemble_config, config_path)
        logger.info(f"Ensemble configuration saved to {config_path}")
    
    def load_models(self, tfidf_path: str, config_path: str) -> bool:
        """
        Load the ensemble model and its components.
        
        Args:
            tfidf_path: Path to the saved TF-IDF model
            config_path: Path to the saved ensemble configuration
            
        Returns:
            Boolean indicating whether loading was successful
        """
        logger.info(f"Loading ensemble model from {self.models_path}")
        
        try:
            # Load TF-IDF processor
            tfidf_loaded = self.tfidf_processor.load_models(tfidf_path)
            
            # Load ensemble configuration
            ensemble_config = joblib.load(config_path)
            self.weights = ensemble_config.get('weights', self.weights)
            self.insurance_keywords = ensemble_config.get('insurance_keywords', self.insurance_keywords)
            
            logger.info(f"Ensemble model loaded successfully")
            return tfidf_loaded
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # This is just an example - in real implementation you would use your own data
    
    from src.preprocessing.preprocessing import DataPreprocessor
    
    # Initialize the data preprocessor
    data_preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Load and preprocess data
    companies_df, taxonomy_df = data_preprocessor.process_company_and_taxonomy(
        company_file="companies.csv",
        taxonomy_file="insurance_taxonomy.csv"
    )
    
    # Initialize the ensemble classifier
    ensemble_classifier = EnsembleClassifier(
        models_path='models/',
        wordnet_weight=0.3,
        keyword_weight=0.2,
        tfidf_weight=0.5
    )
    
    # Classify companies using the ensemble approach
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=5,
        threshold=0.08,
        company_text_column='combined_features'
    )
    
    # Save results
    classified_companies.to_csv('data/processed/ensemble_classified_companies.csv', index=False)
    
    # Save models for later use
    ensemble_classifier.save_models()