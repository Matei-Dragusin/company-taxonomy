"""
Module for implementing an Ensemble Classifier for insurance taxonomy,
combining TF-IDF with other approaches. This is an optimized version
with better performance for large datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Set
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
import time
from functools import lru_cache

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedEnsembleClassifier:
    """
    Optimized Ensemble Classifier that combines multiple approaches to
    classify companies into the insurance taxonomy.
    """
    
    def __init__(self, models_path: str = 'models/', 
                 wordnet_weight: float = 0.25,
                 keyword_weight: float = 0.25,
                 tfidf_weight: float = 0.5,
                 synonym_cache_size: int = 1024):
        """
        Initialize the Ensemble classifier with optimization parameters.
        
        Args:
            models_path: Path for saving/loading models
            wordnet_weight: Weight for the WordNet-based approach
            keyword_weight: Weight for the keyword-based approach
            tfidf_weight: Weight for the TF-IDF approach
            synonym_cache_size: Size of the LRU cache for synonyms
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
        
        # Configure WordNet synonym caching
        self.synonym_cache_size = synonym_cache_size
        self._generate_synonyms = lru_cache(maxsize=synonym_cache_size)(self._generate_synonyms_impl)
        
        # Store preprocessed data for quicker access
        self.taxonomy_key_terms_cache = {}
        self.taxonomy_synonyms_cache = {}
        self.taxonomy_sectors_cache = {}
        
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
        
        # Build sector keyword sets for faster matching
        self.sector_keywords = {}
        for sector, keywords in self.insurance_keywords.items():
            self.sector_keywords[sector] = set(keywords)
        
        # Ensure we have NLTK WordNet available
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download WordNet: {e}")
    
    def preprocess_taxonomy(self, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess taxonomy data for classification.
        Optimized to avoid redundant processing.
        
        Args:
            taxonomy_df: DataFrame with insurance taxonomy
            
        Returns:
            DataFrame with preprocessed taxonomy
        """
        logger.info("Preprocessing taxonomy for the Ensemble classifier")
        start_time = time.time()
        
        df_result = taxonomy_df.copy()
        
        # Extract key terms from each label
        if 'cleaned_label' in df_result.columns:
            # Process in batches to avoid memory issues
            key_terms_list = []
            synonyms_list = []
            sectors_list = []
            
            # Use numeric index for faster lookups
            for idx, row in df_result.iterrows():
                label_id = idx
                cleaned_label = row['cleaned_label']
                
                # Process key terms
                if label_id in self.taxonomy_key_terms_cache:
                    key_terms = self.taxonomy_key_terms_cache[label_id]
                else:
                    key_terms = self._extract_key_terms(cleaned_label)
                    self.taxonomy_key_terms_cache[label_id] = key_terms
                key_terms_list.append(key_terms)
                
                # Process synonyms - limit to important terms only (max 10) to improve performance
                if label_id in self.taxonomy_synonyms_cache:
                    synonyms = self.taxonomy_synonyms_cache[label_id]
                else:
                    # Only generate synonyms for important terms to improve performance
                    important_terms = key_terms[:min(10, len(key_terms))]
                    synonyms = self._generate_synonyms(tuple(important_terms))
                    self.taxonomy_synonyms_cache[label_id] = synonyms
                synonyms_list.append(synonyms)
                
                # Process insurance sectors
                if label_id in self.taxonomy_sectors_cache:
                    sectors = self.taxonomy_sectors_cache[label_id]
                else:
                    sectors = self._map_to_insurance_sectors(cleaned_label)
                    self.taxonomy_sectors_cache[label_id] = sectors
                sectors_list.append(sectors)
            
            # Add the processed data to the DataFrame
            df_result['key_terms'] = key_terms_list
            df_result['synonyms'] = synonyms_list
            df_result['insurance_sectors'] = sectors_list
        
        logger.info(f"Taxonomy preprocessing complete in {time.time() - start_time:.2f} seconds")
        
        return df_result
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from a text. Optimized for performance.
        
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
        
        # Add bigrams (phrases of 2 words) - only add the most important ones
        # This reduces the computational load while preserving the most important information
        if len(words) > 1:
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:  # Only add meaningful bigrams
                    key_terms.append(f"{words[i]} {words[i+1]}")
        
        return key_terms
    
    def _generate_synonyms_impl(self, terms: tuple) -> List[str]:
        """
        Generate synonyms for given terms using WordNet.
        This is the implementation - wrapped with lru_cache decorator.
        
        Args:
            terms: Tuple of terms to generate synonyms for (tuple for hashability)
            
        Returns:
            Extended list with original terms and their synonyms
        """
        # Convert tuple back to list
        terms_list = list(terms)
        all_terms = list(terms_list)  # Start with original terms
        
        # Limit the number of terms we process to avoid performance issues
        for term in terms_list[:min(5, len(terms_list))]:  # Only process up to 5 terms
            # Skip phrases, use only single words
            if ' ' in term:
                continue
                
            # Look for synonyms in WordNet - limit to first 2 synsets for performance
            for synset in list(wordnet.synsets(term))[:2]:  # Only check first 2 synsets
                for lemma in list(synset.lemmas())[:3]:  # Only check first 3 lemmas
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and synonym not in all_terms:
                        all_terms.append(synonym)
                        # Limit to max 10 synonyms per term for performance
                        if len(all_terms) >= len(terms_list) + 10:
                            break
                if len(all_terms) >= len(terms_list) + 10:
                    break
            if len(all_terms) >= len(terms_list) + 10:
                break
        
        return all_terms
    
    def _map_to_insurance_sectors(self, text: str) -> List[str]:
        """
        Map a text to relevant insurance sectors based on keywords.
        Optimized for performance.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relevant insurance sectors
        """
        if not isinstance(text, str):
            return []
            
        relevant_sectors = []
        text_lower = text.lower()
        
        # Use set operations for faster matching
        text_words = set(text_lower.split())
        
        for sector, keywords_set in self.sector_keywords.items():
            # Check if any keyword is in the text
            # First try simple word matching for performance
            if any(keyword in text_words for keyword in keywords_set if ' ' not in keyword):
                relevant_sectors.append(sector)
                continue
                
            # Then check phrases
            for keyword in keywords_set:
                if ' ' in keyword and keyword in text_lower:
                    relevant_sectors.append(sector)
                    break
        
        return relevant_sectors
    
    def compute_keyword_similarity(self, company_text: str, 
                                  taxonomy_df: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity based on insurance keyword matching.
        Optimized for performance.
        
        Args:
            company_text: Preprocessed company text
            taxonomy_df: Preprocessed taxonomy DataFrame
            
        Returns:
            Array of similarity scores for each taxonomy label
        """
        if not isinstance(company_text, str):
            company_text = ""
            
        company_sectors = self._map_to_insurance_sectors(company_text)
        company_sectors_set = set(company_sectors)
        
        similarities = []
        
        # Use vectorized operations where possible
        for sectors in taxonomy_df['insurance_sectors']:
            sectors_set = set(sectors)
            # Calculate Jaccard similarity
            if not company_sectors_set or not sectors_set:
                similarities.append(0.0)
            else:
                intersection = len(company_sectors_set & sectors_set)
                union = len(company_sectors_set | sectors_set)
                similarities.append(intersection / union if union > 0 else 0.0)
        
        return np.array(similarities)
    
    def compute_wordnet_similarity(self, company_text: str,
                                  taxonomy_df: pd.DataFrame) -> np.ndarray:
        """
        Compute similarity based on WordNet synonyms matching.
        Optimized for performance.
        
        Args:
            company_text: Preprocessed company text
            taxonomy_df: Preprocessed taxonomy DataFrame
            
        Returns:
            Array of similarity scores for each taxonomy label
        """
        if not isinstance(company_text, str):
            company_text = ""
            
        # Extract key terms from company text - limit to most important ones
        company_terms = self._extract_key_terms(company_text)[:10]  # Limit to top 10 terms
        
        # Generate synonyms for company terms
        company_synonyms = self._generate_synonyms(tuple(company_terms))
        company_synonyms_set = set(company_synonyms)
        
        similarities = []
        
        # Use set operations for faster matching
        for synonyms in taxonomy_df['synonyms']:
            synonyms_set = set(synonyms)
            
            # Calculate similarity based on term overlap
            if not company_synonyms_set or not synonyms_set:
                similarities.append(0.0)
            else:
                common_terms = len(company_synonyms_set & synonyms_set)
                denominator = (len(company_synonyms_set) + len(synonyms_set) - common_terms)
                similarity = common_terms / denominator if denominator > 0 else 0.0
                similarities.append(similarity)
        
        return np.array(similarities)
    
    def ensemble_classify(self, companies_df: pd.DataFrame, 
                         taxonomy_df: pd.DataFrame,
                         top_k: int = 5, 
                         threshold: float = 0.08,
                         company_text_column: str = 'combined_features',
                         batch_size: int = 100) -> pd.DataFrame:
        """
        Classify companies using the ensemble approach.
        Optimized with batch processing for better performance with large datasets.
        
        Args:
            companies_df: DataFrame with preprocessed company data
            taxonomy_df: DataFrame with preprocessed taxonomy data
            top_k: Maximum number of labels to assign to a company
            threshold: Minimum similarity threshold to assign a label
            company_text_column: Column containing text for comparison
            batch_size: Number of companies to process in each batch
            
        Returns:
            DataFrame with added classification results
        """
        total_companies = len(companies_df)
        logger.info(f"Starting optimized ensemble classification for {total_companies} companies")
        start_time = time.time()
        
        # Preprocess taxonomy for ensemble methods
        enhanced_taxonomy_df = self.preprocess_taxonomy(taxonomy_df)
        
        # First get TF-IDF classification and similarity matrix
        logger.info("Computing TF-IDF vectors and similarities")
        tfidf_start = time.time()
        company_vectors = self.tfidf_processor.transform_company_data(companies_df, company_text_column)
        taxonomy_vectors = self.tfidf_processor.transform_taxonomy(taxonomy_df, 'cleaned_label')
        
        # Create the final results DataFrame
        result_df = companies_df.copy()
        all_ensemble_matches = []
        all_ensemble_scores = []
        
        # Process in batches to reduce memory usage
        num_batches = (total_companies + batch_size - 1) // batch_size
        logger.info(f"Processing {total_companies} companies in {num_batches} batches of size {batch_size}")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_companies)
            
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} (companies {batch_start}-{batch_end})")
            batch_start_time = time.time()
            
            # Get batch of company vectors
            batch_vectors = company_vectors[batch_start:batch_end]
            
            # Calculate TF-IDF similarity for the batch
            batch_tfidf_similarities = cosine_similarity(batch_vectors, taxonomy_vectors)
            
            # Process each company in the batch
            for i in range(batch_end - batch_start):
                company_idx = batch_start + i
                company = companies_df.iloc[company_idx]
                company_text = company[company_text_column] if company_text_column in company.index else ""
                
                # Get TF-IDF similarity for this company
                company_tfidf_sim = batch_tfidf_similarities[i]
                
                # Calculate other similarities only if needed (if TF-IDF weight < 1.0)
                if self.weights['keyword'] > 0:
                    company_keyword_sim = self.compute_keyword_similarity(company_text, enhanced_taxonomy_df)
                else:
                    company_keyword_sim = np.zeros(len(enhanced_taxonomy_df))
                    
                if self.weights['wordnet'] > 0:
                    company_wordnet_sim = self.compute_wordnet_similarity(company_text, enhanced_taxonomy_df)
                else:
                    company_wordnet_sim = np.zeros(len(enhanced_taxonomy_df))
                
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
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx+1} processed in {batch_time:.2f} seconds ({(batch_end-batch_start)/batch_time:.1f} companies/second)")
        
        # Add results to DataFrame
        result_df['insurance_labels'] = all_ensemble_matches
        result_df['insurance_label_scores'] = all_ensemble_scores
        
        # Create comma-separated string of labels
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        total_time = time.time() - start_time
        logger.info(f"Ensemble classification complete in {total_time:.2f} seconds ({total_companies/total_time:.1f} companies/second)")
        
        # Clear caches to free memory
        self._generate_synonyms.cache_clear()
        
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
            'insurance_keywords': self.insurance_keywords,
            'synonym_cache_size': self.synonym_cache_size
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
            
            # Build sector keyword sets
            self.sector_keywords = {}
            for sector, keywords in self.insurance_keywords.items():
                self.sector_keywords[sector] = set(keywords)
            
            # Update cache size if needed
            new_cache_size = ensemble_config.get('synonym_cache_size', self.synonym_cache_size)
            if new_cache_size != self.synonym_cache_size:
                self.synonym_cache_size = new_cache_size
                # Re-wrap the function with the new cache size
                self._generate_synonyms = lru_cache(maxsize=new_cache_size)(self._generate_synonyms_impl)
            
            logger.info(f"Ensemble model loaded successfully")
            return tfidf_loaded
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False


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
    
    # Initialize the optimized ensemble classifier
    ensemble_classifier = OptimizedEnsembleClassifier(
        models_path='models/',
        wordnet_weight=0.25,
        keyword_weight=0.25,
        tfidf_weight=0.5,
        synonym_cache_size=1024
    )
    
    # Classify companies using the ensemble approach
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=5,
        threshold=0.08,
        company_text_column='combined_features',
        batch_size=100
    )
    
    # Save results
    classified_companies.to_csv('data/processed/ensemble_classified_companies.csv', index=False)
    
    # Save models for later use
    ensemble_classifier.save_models()