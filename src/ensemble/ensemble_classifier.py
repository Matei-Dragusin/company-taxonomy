"""
Consolidated module for implementing the ensemble classifier for insurance taxonomy.
This module combines the functionality of both basic and optimized classifiers.
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
import time
from functools import lru_cache
import traceback

# Import TFIDFProcessor
from src.feature_engineering.tfidf_processor import TFIDFProcessor

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleClassifier:
    """
    Consolidated ensemble classifier that combines multiple approaches for
    classifying companies into insurance taxonomy. Includes both basic
    functionality and optimizations for large datasets.
    """
    
    def __init__(self, models_path: str = 'models/', 
                 wordnet_weight: float = 0.25,
                 keyword_weight: float = 0.25,
                 tfidf_weight: float = 0.5,
                 optimizer_mode: bool = True,
                 synonym_cache_size: int = 1024):
        """
        Initialize the ensemble classifier.
        
        Args:
            models_path: Path for saving/loading models
            wordnet_weight: Weight for WordNet-based approach
            keyword_weight: Weight for keyword-based approach
            tfidf_weight: Weight for TF-IDF approach
            optimizer_mode: Whether to use optimizations for large datasets
            synonym_cache_size: Size of LRU cache for synonyms (optimized mode only)
        """
        self.models_path = models_path
        self.optimizer_mode = optimizer_mode
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            logger.info(f"Created directory {models_path}")
        
        # Weights for combining different approaches
        self.weights = {
            'tfidf': tfidf_weight,
            'wordnet': wordnet_weight,
            'keyword': keyword_weight
        }
        
        # Normalize weights
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Initialize TF-IDF processor
        self.tfidf_processor = TFIDFProcessor(
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 3),
            models_path=models_path
        )
        
        # Settings specific to optimized mode
        if optimizer_mode:
            self.synonym_cache_size = synonym_cache_size
            # Use lru_cache decorator for synonym generation
            self._generate_synonyms_cached = lru_cache(maxsize=synonym_cache_size)(self._generate_synonyms_impl)
            
            # Cache for preprocessed data for faster access
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
        
        # Build sector keyword sets for faster matching (for optimized mode)
        if optimizer_mode:
            self.sector_keywords = {}
            for sector, keywords in self.insurance_keywords.items():
                self.sector_keywords[sector] = set(keywords)
        
        # Ensure we have NLTK WordNet available
        try:
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download WordNet: {e}")
    
    def get_exact_match_bonus(self, company_text: str, taxonomy_label: str) -> float:
        """
        Add bonus for exact matches of important terms.
        
        Args:
            company_text: Company text
            taxonomy_label: Taxonomy label text
            
        Returns:
            Bonus score for exact matches
        """
        if not isinstance(company_text, str) or not isinstance(taxonomy_label, str):
            return 0.0
            
        # Convert to lowercase
        company_text = company_text.lower()
        taxonomy_label = taxonomy_label.lower()
        
        # Check for exact term matches
        taxonomy_terms = set(taxonomy_label.split())
        company_terms = set(company_text.split())
        exact_matches = taxonomy_terms.intersection(company_terms)
        
        # Calculate bonus based on matches
        if len(taxonomy_terms) > 0:
            match_ratio = len(exact_matches) / len(taxonomy_terms)
            return match_ratio * 0.3  # Up to 0.3 bonus for exact matches
        return 0.0
    
    # Common methods for both modes
    
    def preprocess_taxonomy(self, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess taxonomy data for classification.
        Optimized to avoid redundant processing in optimized mode.
        
        Args:
            taxonomy_df: DataFrame with insurance taxonomy
            
        Returns:
            DataFrame with preprocessed taxonomy
        """
        logger.info("Preprocessing taxonomy for the ensemble classifier")
        start_time = time.time()
        
        # Create a copy to avoid modifying the original
        df_result = taxonomy_df.copy()
        
        # Extract key terms from each label
        if 'cleaned_label' in df_result.columns:
            if self.optimizer_mode:
                # Process in batches to avoid memory issues
                key_terms_list = []
                synonyms_list = []
                sectors_list = []
                
                # Use numeric index for faster lookups
                for idx, row in df_result.iterrows():
                    label_id = idx
                    cleaned_label = str(row['cleaned_label']) if pd.notnull(row['cleaned_label']) else ""
                    
                    # Process key terms
                    if label_id in self.taxonomy_key_terms_cache:
                        key_terms = self.taxonomy_key_terms_cache[label_id]
                    else:
                        key_terms = self._extract_key_terms(cleaned_label)
                        self.taxonomy_key_terms_cache[label_id] = key_terms
                    key_terms_list.append(key_terms)
                    
                    # Process synonyms - limit to important terms only
                    if label_id in self.taxonomy_synonyms_cache:
                        synonyms = self.taxonomy_synonyms_cache[label_id]
                    else:
                        # Only generate synonyms for important terms to improve performance
                        important_terms = key_terms[:min(10, len(key_terms))]
                        # Convert to tuple for hashability in lru_cache
                        important_terms_tuple = tuple(important_terms)
                        synonyms = self._generate_synonyms_cached(important_terms_tuple)
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
            else:
                # Basic mode processing - simpler approach
                df_result['key_terms'] = df_result['cleaned_label'].apply(self._extract_key_terms)
                df_result['synonyms'] = df_result['key_terms'].apply(self._generate_synonyms)
                df_result['insurance_sectors'] = df_result['cleaned_label'].apply(self._map_to_insurance_sectors)
        
        processing_time = time.time() - start_time
        logger.info(f"Taxonomy preprocessing complete in {processing_time:.2f} seconds")
        
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
        
        # Add bigrams if in optimized mode, otherwise add n-grams up to 3
        if self.optimizer_mode:
            # Add only bigrams for important words to optimize performance
            if len(words) > 1:
                for i in range(len(words) - 1):
                    if len(words[i]) > 3 and len(words[i+1]) > 3:
                        key_terms.append(f"{words[i]} {words[i+1]}")
        else:
            # Add all n-grams in basic mode
            for n in range(2, 4):  # Generate bigrams and trigrams
                for i in range(len(words) - n + 1):
                    key_terms.append(' '.join(words[i:i+n]))
        
        return key_terms
    
    def _generate_synonyms_impl(self, terms_tuple: tuple) -> List[str]:
        """
        Generate synonyms for given terms using WordNet.
        Used with lru_cache in optimized mode.
        
        Args:
            terms_tuple: Tuple of terms to generate synonyms for
            
        Returns:
            Extended list with original terms and their synonyms
        """
        # Convert tuple back to list
        terms_list = list(terms_tuple)
        all_terms = list(terms_list)  # Start with original terms
        
        # Limit the number of terms processed for performance
        for term in terms_list[:min(5, len(terms_list))]:
            # Skip phrases, use only single words
            if ' ' in term:
                continue
                
            # Look for synonyms in WordNet with limits
            try:
                synsets = list(wordnet.synsets(term))[:2]  # Only first 2 synsets
                
                for synset in synsets:
                    lemmas = list(synset.lemmas())[:3]  # Only first 3 lemmas
                    
                    for lemma in lemmas:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != term and synonym not in all_terms:
                            all_terms.append(synonym)
                            # Limit synonyms per term
                            if len(all_terms) >= len(terms_list) + 10:
                                break
                    if len(all_terms) >= len(terms_list) + 10:
                        break
                if len(all_terms) >= len(terms_list) + 10:
                    break
            except Exception as e:
                logger.warning(f"Error generating synonyms for term '{term}': {e}")
                continue
        
        return all_terms
    
    def _generate_synonyms(self, terms: List[str]) -> List[str]:
        """
        Generate synonyms for given terms.
        Core implementation used in basic mode or as a public interface in optimized mode.
        
        Args:
            terms: List of terms to generate synonyms for
            
        Returns:
            Extended list with original terms and their synonyms
        """
        if self.optimizer_mode:
            # Convert list to tuple for caching
            terms_tuple = tuple(terms)
            return self._generate_synonyms_cached(terms_tuple)
        
        # Basic implementation for non-optimized mode
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
            
        text_lower = text.lower()
        
        if self.optimizer_mode:
            # Optimized implementation with set operations
            relevant_sectors = []
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
        else:
            # Basic implementation
            relevant_sectors = []
            
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
            
        company_sectors = self._map_to_insurance_sectors(company_text)
        
        if self.optimizer_mode:
            # Optimized implementation with set operations
            company_sectors_set = set(company_sectors)
            similarities = []
            
            for sectors in taxonomy_df['insurance_sectors']:
                sectors_set = set(sectors)
                # Calculate Jaccard similarity
                if not company_sectors_set or not sectors_set:
                    similarities.append(0.0)
                else:
                    intersection = len(company_sectors_set & sectors_set)
                    union = len(company_sectors_set | sectors_set)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    # Add a boost to high scores
                    if similarity > 0.5:  # If it's already a good match
                        similarity = 0.5 + (similarity * 0.5)  # Boost the higher end
                        
                    similarities.append(similarity)
        else:
            # Basic implementation
            similarities = []
            
            for _, row in taxonomy_df.iterrows():
                label_sectors = row.get('insurance_sectors', [])
                
                # Calculate similarity
                if not company_sectors or not label_sectors:
                    similarities.append(0.0)
                else:
                    intersection = len(set(company_sectors) & set(label_sectors))
                    union = len(set(company_sectors) | set(label_sectors))
                    similarity = intersection / union if union > 0 else 0.0
                    
                    # Add a boost to high scores
                    if similarity > 0.5:  # If it's already a good match
                        similarity = 0.5 + (similarity * 0.5)  # Boost the higher end
                        
                    similarities.append(similarity)
        
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
        
        if self.optimizer_mode:
            # Optimized implementation with limit on terms
            company_terms = company_terms[:10]  # Limit to top 10 terms
            
        # Generate synonyms for company terms
        company_synonyms = self._generate_synonyms(company_terms)
        company_synonyms_set = set(company_synonyms)
        
        similarities = []
        
        # Calculate similarity with each taxonomy label
        for synonyms in taxonomy_df['synonyms']:
            synonyms_set = set(synonyms)
            
            # Calculate similarity based on term overlap
            if not company_synonyms_set or not synonyms_set:
                similarities.append(0.0)
            else:
                common_terms = len(company_synonyms_set & synonyms_set)
                denominator = (len(company_synonyms_set) + len(synonyms_set) - common_terms)
                similarity = common_terms / denominator if denominator > 0 else 0.0
                
                # Add a boost to high scores
                if similarity > 0.5:  # If it's already a good match
                    similarity = 0.5 + (similarity * 0.5)  # Boost the higher end
                    
                similarities.append(similarity)
        
        return np.array(similarities)
    
    def ensemble_classify(self, companies_df: pd.DataFrame, 
                         taxonomy_df: pd.DataFrame,
                         top_k: int = 5, 
                         threshold: float = 0.08,
                         company_text_column: str = 'combined_features',
                         batch_size: int = 100,
                         ensure_one_tag: bool = True) -> pd.DataFrame:
        """
        Classify companies using the ensemble approach.
        
        Args:
            companies_df: DataFrame with preprocessed company data
            taxonomy_df: DataFrame with preprocessed taxonomy data
            top_k: Maximum number of labels to assign to a company
            threshold: Minimum similarity threshold to assign a label
            company_text_column: Column containing text for comparison
            batch_size: Number of companies to process in each batch (optimized mode only)
            ensure_one_tag: Whether to ensure each company has at least one tag
            
        Returns:
            DataFrame with added classification results
        """
        logger.info(f"Starting {'optimized' if self.optimizer_mode else 'basic'} ensemble classification for {len(companies_df)} companies")
        start_time = time.time()
        
        # Input validation
        if company_text_column not in companies_df.columns:
            raise ValueError(f"Column '{company_text_column}' not found in companies DataFrame")
        
        # Preprocess taxonomy for ensemble methods
        enhanced_taxonomy_df = self.preprocess_taxonomy(taxonomy_df)
        
        # Initialize TF-IDF vectorizer and transform data
        self.tfidf_processor.fit_vectorizer(
            companies_df, 
            taxonomy_df, 
            company_text_column=company_text_column,
            taxonomy_column='cleaned_label'
        )
        
        company_vectors = self.tfidf_processor.transform_company_data(companies_df, company_text_column)
        taxonomy_vectors = self.tfidf_processor.transform_taxonomy(taxonomy_df, 'cleaned_label')
        
        # Create the final results DataFrame
        result_df = companies_df.copy()
        all_ensemble_matches = []
        all_ensemble_scores = []
        
        if self.optimizer_mode:
            # Optimized implementation with batch processing
            total_companies = len(companies_df)
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
                    company_text = str(company[company_text_column]) if pd.notnull(company[company_text_column]) else ""
                    
                    # Get TF-IDF similarity for this company
                    company_tfidf_sim = batch_tfidf_similarities[i]
                    
                    # Calculate other similarities
                    company_keyword_sim = self.compute_keyword_similarity(company_text, enhanced_taxonomy_df)
                    company_wordnet_sim = self.compute_wordnet_similarity(company_text, enhanced_taxonomy_df)
                    
                    # Combine similarities using weighted average
                    ensemble_sim = (
                        self.weights['tfidf'] * company_tfidf_sim +
                        self.weights['keyword'] * company_keyword_sim +
                        self.weights['wordnet'] * company_wordnet_sim
                    )
                    
                    # Add exact match bonus
                    for j, label in enumerate(enhanced_taxonomy_df['label']):
                        exact_match_bonus = self.get_exact_match_bonus(company_text, label)
                        ensemble_sim[j] += exact_match_bonus
                    
                    # Apply power transformation to enhance high values
                    enhanced_sim = np.power(ensemble_sim, 2)  # Square the values to make high values higher
                    # Normalize to [0,1] again
                    if enhanced_sim.max() > 0:
                        enhanced_sim = enhanced_sim / enhanced_sim.max()
                    
                    # Get top matches
                    top_indices = enhanced_sim.argsort()[-top_k:][::-1]
                    
                    # Filter by threshold with adaptive approach
                    company_matches = []
                    company_scores = []
                    
                    # First pass: apply standard threshold
                    for idx in top_indices:
                        score = enhanced_sim[idx]
                        if score >= threshold:
                            company_matches.append(enhanced_taxonomy_df.iloc[idx]['label'])
                            company_scores.append(float(score))
                    
                    # If no labels were assigned and ensure_one_tag is True, find the best alternative
                    if not company_matches and ensure_one_tag:
                        # Try to find a tag based on the highest similarity score
                        if len(top_indices) > 0:
                            best_idx = top_indices[0]  # Index with highest similarity
                            company_matches.append(enhanced_taxonomy_df.iloc[best_idx]['label'])
                            company_scores.append(float(enhanced_sim[best_idx]))
                        else:
                            # As a last resort, try to assign based on sector or niche
                            sector_tag = self._find_tag_by_sector(company, enhanced_taxonomy_df)
                            if sector_tag:
                                company_matches.append(sector_tag[0])
                                company_scores.append(sector_tag[1])
                            else:
                                # If all else fails, assign a generic insurance tag
                                default_tag = "General Insurance"
                                for idx, label in enumerate(enhanced_taxonomy_df['label']):
                                    if "general" in label.lower() and "insurance" in label.lower():
                                        default_tag = label
                                        break
                                company_matches.append(default_tag)
                                company_scores.append(0.1)  # Low confidence score for default tag
                    
                    all_ensemble_matches.append(company_matches)
                    all_ensemble_scores.append(company_scores)
                
                batch_time = time.time() - batch_start_time
                logger.info(f"Batch {batch_idx+1} processed in {batch_time:.2f} seconds")
        else:
            # Basic implementation - process all companies at once
            
            # Calculate TF-IDF similarities
            tfidf_similarities = cosine_similarity(company_vectors, taxonomy_vectors)
            
            # Process each company
            for i, (_, company) in enumerate(companies_df.iterrows()):
                company_text = str(company[company_text_column]) if pd.notnull(company[company_text_column]) else ""
                
                # Get TF-IDF similarity for this company
                company_tfidf_sim = tfidf_similarities[i]
                
                # Calculate other similarities
                company_keyword_sim = self.compute_keyword_similarity(company_text, enhanced_taxonomy_df)
                company_wordnet_sim = self.compute_wordnet_similarity(company_text, enhanced_taxonomy_df)
                
                # Combine similarities using weighted average
                ensemble_sim = (
                    self.weights['tfidf'] * company_tfidf_sim +
                    self.weights['keyword'] * company_keyword_sim +
                    self.weights['wordnet'] * company_wordnet_sim
                )
                
                # Add exact match bonus
                for j, label in enumerate(enhanced_taxonomy_df['label']):
                    exact_match_bonus = self.get_exact_match_bonus(company_text, label)
                    ensemble_sim[j] += exact_match_bonus
                
                # Apply power transformation to enhance high values
                enhanced_sim = np.power(ensemble_sim, 2)  # Square the values to make high values higher
                # Normalize to [0,1] again
                if enhanced_sim.max() > 0:
                    enhanced_sim = enhanced_sim / enhanced_sim.max()
                
                # Get top matches
                top_indices = enhanced_sim.argsort()[-top_k:][::-1]
                
                # Filter by threshold with adaptive approach
                company_matches = []
                company_scores = []
                
                # First pass: apply standard threshold
                for idx in top_indices:
                    score = enhanced_sim[idx]
                    if score >= threshold:
                        company_matches.append(enhanced_taxonomy_df.iloc[idx]['label'])
                        company_scores.append(float(score))
                
                # If no labels were assigned and ensure_one_tag is True, find the best alternative
                if not company_matches and ensure_one_tag:
                    # Try to find a tag based on the highest similarity score
                    if len(top_indices) > 0:
                        best_idx = top_indices[0]  # Index with highest similarity
                        company_matches.append(enhanced_taxonomy_df.iloc[best_idx]['label'])
                        company_scores.append(float(enhanced_sim[best_idx]))
                    else:
                        # As a last resort, try to assign based on sector or niche
                        sector_tag = self._find_tag_by_sector(company, enhanced_taxonomy_df)
                        if sector_tag:
                            company_matches.append(sector_tag[0])
                            company_scores.append(sector_tag[1])
                        else:
                            # If all else fails, assign a generic insurance tag
                            default_tag = "General Insurance"
                            for idx, label in enumerate(enhanced_taxonomy_df['label']):
                                if "general" in label.lower() and "insurance" in label.lower():
                                    default_tag = label
                                    break
                            company_matches.append(default_tag)
                            company_scores.append(0.1)  # Low confidence score for default tag
                
                all_ensemble_matches.append(company_matches)
                all_ensemble_scores.append(company_scores)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} companies")
        
        # Add results to DataFrame
        result_df['insurance_labels'] = all_ensemble_matches
        result_df['insurance_label_scores'] = all_ensemble_scores
        
        # Create comma-separated string of labels
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        # Clear caches if in optimized mode
        if self.optimizer_mode:
            self._generate_synonyms_cached.cache_clear()
        
        total_time = time.time() - start_time
        logger.info(f"Ensemble classification complete in {total_time:.2f} seconds")
        
        return result_df

    def _find_tag_by_sector(self, company: pd.Series, taxonomy_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Find the most appropriate tag based on company sector or niche.
        
        Args:
            company: Company data series
            taxonomy_df: Taxonomy DataFrame
            
        Returns:
            Tuple containing the best matching tag and a confidence score
        """
        # Check if we have sector or niche information
        sector = company.get('sector', "") if pd.notnull(company.get('sector', "")) else ""
        niche = company.get('niche', "") if pd.notnull(company.get('niche', "")) else ""
        category = company.get('category', "") if pd.notnull(company.get('category', "")) else ""
        
        # Combine sector, category and niche
        sector_info = f"{sector} {category} {niche}".lower()
        
        if not sector_info.strip():
            return None
        
        # Look for relevant tags
        best_match = None
        best_score = 0
        
        for _, row in taxonomy_df.iterrows():
            label = row['label']
            label_lower = label.lower()
            
            # Simple keyword matching
            score = 0
            
            # Check if any words from the sector appear in the label
            for word in sector_info.split():
                if len(word) > 3 and word in label_lower:
                    score += 0.2
            
            # Check if sector is directly related to label
            for key_sector in self.insurance_keywords:
                if any(keyword in sector_info for keyword in self.insurance_keywords[key_sector]):
                    sector_tags = [tag.lower() for tag in self.insurance_keywords[key_sector]]
                    if any(tag in label_lower for tag in sector_tags):
                        score += 0.3
            
            if score > best_score:
                best_score = score
                best_match = label
        
        # If no good match found based on sector, return None
        if best_score < 0.2:
            return None
        
        return (best_match, best_score)
    
    def save_models(self, filename_prefix: str = 'ensemble_model') -> None:
        """
        Save the ensemble model and its components.
        
        Args:
            filename_prefix: Prefix for the filename when saving
        """
        logger.info(f"Saving ensemble model to {self.models_path}")
        
        try:
            # Save TF-IDF processor
            self.tfidf_processor.save_models(filename_prefix='tfidf_component')
            
            # Save ensemble configuration
            ensemble_config = {
                'weights': self.weights,
                'insurance_keywords': self.insurance_keywords,
                'optimizer_mode': self.optimizer_mode
            }
            
            if self.optimizer_mode:
                ensemble_config['synonym_cache_size'] = self.synonym_cache_size
            
            config_path = os.path.join(self.models_path, f"{filename_prefix}_config.joblib")
            joblib.dump(ensemble_config, config_path)
            logger.info(f"Ensemble configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
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
            self.optimizer_mode = ensemble_config.get('optimizer_mode', self.optimizer_mode)
            
            # Build sector keyword sets
            if self.optimizer_mode:
                self.sector_keywords = {}
                for sector, keywords in self.insurance_keywords.items():
                    self.sector_keywords[sector] = set(keywords)
                
                # Update cache size if needed
                new_cache_size = ensemble_config.get('synonym_cache_size', self.synonym_cache_size)
                if new_cache_size != self.synonym_cache_size:
                    self.synonym_cache_size = new_cache_size
                    # Re-wrap the function with the new cache size
                    self._generate_synonyms_cached = lru_cache(maxsize=new_cache_size)(self._generate_synonyms_impl)
            
            logger.info(f"Ensemble model loaded successfully")
            return tfidf_loaded
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return False


    def export_description_label_csv(self, classified_df: pd.DataFrame, 
                               output_path: str = 'data/processed/description_label_results.csv',
                               description_column: str = 'description',
                                include_scores: bool = False) -> None:
        """
        Export a CSV file with company descriptions and their assigned insurance labels.
        
        Args:
            classified_df: DataFrame with classification results
            output_path: Path for the output CSV file
            description_column: Column name containing company descriptions
            include_scores: Whether to include confidence scores in the output
        """
        logger.info(f"Exporting description-label CSV to {output_path}")
        
        if description_column not in classified_df.columns:
            logger.error(f"Description column '{description_column}' not found in DataFrame")
            raise ValueError(f"Description column '{description_column}' not found")
        
        export_columns = {
            'description': classified_df[description_column],
            'insurance_label': classified_df['insurance_label']
        }
        
        # Include additional information if available and useful
        if 'sector' in classified_df.columns:
            export_columns['sector'] = classified_df['sector']
        
        if 'niche' in classified_df.columns:
            export_columns['niche'] = classified_df['niche']
        
        if 'category' in classified_df.columns:
            export_columns['category'] = classified_df['category']
            
        if include_scores and 'insurance_label_scores' in classified_df.columns:
            # Convert score lists to strings for CSV export
            export_columns['label_scores'] = classified_df['insurance_label_scores'].apply(
                lambda x: ', '.join([f"{score:.4f}" for score in x]) if isinstance(x, list) else x
            )
        
        # Create a new DataFrame with only the selected columns
        export_df = pd.DataFrame(export_columns)
        
        # Save to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(export_df)} records to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example usage of the ensemble classifier
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
        tfidf_weight=0.5,
        optimizer_mode=True
    )
    
    # Classify companies using the ensemble approach
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=5,
        threshold=0.08,
        company_text_column='combined_features'
    )
    
    # Save full results
    classified_companies.to_csv('data/processed/ensemble_classified_companies.csv', index=False)
    
    # Save simplified description-label results
    ensemble_classifier.export_description_label_csv(
        classified_companies,
        output_path='data/processed/description_label_results.csv',
        description_column='description'
    )
    
    # Save models for future use
    ensemble_classifier.save_models()