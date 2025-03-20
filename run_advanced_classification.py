#!/usr/bin/env python
"""
Script to run advanced classification methods (SBERT and Adaptive Ensemble)
for the insurance taxonomy classification task.
"""

import os
import sys
import pandas as pd
import logging
import argparse
import time
from typing import Tuple

# Add the project root directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure necessary directories exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def load_and_preprocess_data(company_file: str, taxonomy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess company and taxonomy data.
    
    Args:
        company_file: Name of the company file
        taxonomy_file: Name of the taxonomy file
        
    Returns:
        Tuple with preprocessed company and taxonomy DataFrames
    """
    logger.info(f"Loading and preprocessing data from {company_file} and {taxonomy_file}")
    start_time = time.time()
    
    # Import preprocessing module
    try:
        from src.preprocessing.preprocessing import DataPreprocessor
    except ImportError as e:
        logger.error(f"Error importing DataPreprocessor: {e}")
        raise
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Process data
    try:
        companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
            company_file=company_file,
            taxonomy_file=taxonomy_file,
            output_company_file="processed_companies.csv",
            output_taxonomy_file="processed_taxonomy.csv"
        )
        
        logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
        return companies_df, taxonomy_df
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def run_sbert_classification(companies_df: pd.DataFrame, 
                           taxonomy_df: pd.DataFrame, 
                           top_k: int = 3, 
                           threshold: float = 0.08,
                           model_name: str = 'all-MiniLM-L6-v2',
                           output_file: str = "sbert_classified_companies.csv") -> pd.DataFrame:
    """
    Run classification using SBERT embeddings.
    
    Args:
        companies_df: DataFrame with preprocessed company data
        taxonomy_df: DataFrame with preprocessed taxonomy data
        top_k: Maximum number of labels to assign
        threshold: Minimum similarity threshold
        model_name: Name of the SBERT model to use
        output_file: Name for the output file
        
    Returns:
        DataFrame with classification results
    """
    logger.info(f"Starting SBERT classification with model: {model_name}")
    start_time = time.time()
    
    # Import SBERT processor
    try:
        from src.feature_engineering.sbert_processor import SBERTProcessor
    except ImportError as e:
        logger.error(f"Error importing SBERTProcessor: {e}")
        raise
    
    # Initialize SBERT processor
    try:
        sbert_processor = SBERTProcessor(
            model_name=model_name,
            models_path='models/'
        )
        logger.info("SBERT processor initialized")
    except Exception as e:
        logger.error(f"Error initializing SBERT processor: {e}")
        raise
    
    # Run classification
    try:
        classified_companies = sbert_processor.assign_labels(
            companies_df,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features'
        )
        
        logger.info(f"SBERT classification completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during SBERT classification: {e}")
        raise
    
    # Save results
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"SBERT classification results saved to {output_path}")
    
    return classified_companies

def run_adaptive_ensemble_classification(companies_df: pd.DataFrame, 
                                       taxonomy_df: pd.DataFrame, 
                                       top_k: int = 3, 
                                       threshold: float = 0.08,
                                       use_sbert: bool = True,
                                       sbert_model: str = 'all-MiniLM-L6-v2',
                                       train_meta_model: bool = True,
                                       meta_model_samples: int = 100,
                                       output_file: str = "adaptive_ensemble_classified_companies.csv") -> pd.DataFrame:
    """
    Run classification using the adaptive ensemble approach.
    
    Args:
        companies_df: DataFrame with preprocessed company data
        taxonomy_df: DataFrame with preprocessed taxonomy data
        top_k: Maximum number of labels to assign
        threshold: Minimum similarity threshold
        use_sbert: Whether to use SBERT embeddings
        sbert_model: Name of the SBERT model to use
        train_meta_model: Whether to train the meta-model
        meta_model_samples: Number of samples for meta-model training
        output_file: Name for the output file
        
    Returns:
        DataFrame with classification results
    """
    logger.info(f"Starting adaptive ensemble classification")
    start_time = time.time()
    
    # Import adaptive ensemble classifier
    try:
        from src.ensemble.adaptive_ensemble import AdaptiveEnsembleClassifier
    except ImportError as e:
        logger.error(f"Error importing AdaptiveEnsembleClassifier: {e}")
        raise
    
    # Initialize adaptive ensemble classifier
    try:
        classifier = AdaptiveEnsembleClassifier(
            models_path='models/',
            use_sbert=use_sbert,
            sbert_model_name=sbert_model
        )
        logger.info("Adaptive ensemble classifier initialized")
    except Exception as e:
        logger.error(f"Error initializing adaptive ensemble classifier: {e}")
        raise
    
    # Run classification
    try:
        classified_companies = classifier.ensemble_classify(
            companies_df,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features',
            train_meta_model=train_meta_model,
            meta_model_samples=meta_model_samples
        )
        
        logger.info(f"Adaptive ensemble classification completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during adaptive ensemble classification: {e}")
        raise
    
    # Save results and model
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    classifier.save_model()
    logger.info(f"Adaptive ensemble classification results saved to {output_path}")
    
    return classified_companies

def evaluate_results(classified_df: pd.DataFrame) -> None:
    """
    Print evaluation metrics for classification results.
    
    Args:
        classified_df: DataFrame with classification results
    """
    total_companies = len(classified_df)
    
    # Count companies with labels
    companies_with_labels = 0
    total_labels = 0
    label_counts = {}
    
    for _, row in classified_df.iterrows():
        labels = row['insurance_labels']
        
        # Convert to list if stored as string
        if isinstance(labels, str):
            if labels.startswith('[') and labels.endswith(']'):
                try:
                    labels = eval(labels)
                except:
                    labels = []
            elif labels == 'Unclassified':
                labels = []
            else:
                labels = [label.strip() for label in labels.split(',')]
        
        # Count labels
        if labels and len(labels) > 0:
            companies_with_labels += 1
            total_labels += len(labels)
            
            # Count individual labels
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate metrics
    coverage = (companies_with_labels / total_companies) * 100 if total_companies > 0 else 0
    avg_labels = total_labels / total_companies if total_companies > 0 else 0
    unique_labels = len(label_counts)
    
    # Print metrics
    print("\n=== CLASSIFICATION RESULTS ===")
    print(f"Total companies: {total_companies}")
    print(f"Companies with labels: {companies_with_labels} ({coverage:.2f}%)")
    print(f"Average labels per company: {avg_labels:.2f}")
    print(f"Unique labels used: {unique_labels}")
    
    # Print top 10 most common labels
    print("\n== TOP 10 MOST COMMON LABELS ==")
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for label, count in sorted_labels:
        print(f"{label}: {count}")

def main():
    """Main function to run advanced classification methods"""
    parser = argparse.ArgumentParser(description='Run advanced classification methods for insurance taxonomy')
    
    parser.add_argument('--method', type=str, choices=['sbert', 'adaptive', 'both'], default='both',
                        help='Classification method to use (default: both)')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Name of the company file (default: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Name of the taxonomy file (default: insurance_taxonomy.csv)')
    
    parser.add_argument('--top-k', type=int, default=3,
                        help='Maximum number of labels to assign (default: 3)')
    
    parser.add_argument('--threshold', type=float, default=0.08,
                        help='Minimum similarity threshold (default: 0.08)')
    
    parser.add_argument('--sbert-model', type=str, default='all-MiniLM-L6-v2',
                        help='SBERT model to use (default: all-MiniLM-L6-v2)')
    
    parser.add_argument('--train-meta-model', action='store_true',
                        help='Train meta-model for adaptive ensemble')
    
    parser.add_argument('--meta-model-samples', type=int, default=100,
                        help='Number of samples for meta-model training (default: 100)')
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing and use existing files')
    
    args = parser.parse_args()
    
    logger.info("Starting advanced classification for insurance taxonomy")
    start_time = time.time()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Load and preprocess data if needed
    if args.skip_preprocessing:
        logger.info("Skipping preprocessing, loading processed data")
        processed_path = 'data/processed/'
        
        try:
            companies_df = pd.read_csv(os.path.join(processed_path, 'processed_companies.csv'))
            taxonomy_df = pd.read_csv(os.path.join(processed_path, 'processed_taxonomy.csv'))
            logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return
    else:
        try:
            companies_df, taxonomy_df = load_and_preprocess_data(args.company_file, args.taxonomy_file)
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return
    
    # Run selected classification method(s)
    if args.method in ['sbert', 'both']:
        logger.info("Running SBERT classification")
        try:
            sbert_results = run_sbert_classification(
                companies_df,
                taxonomy_df,
                top_k=args.top_k,
                threshold=args.threshold,
                model_name=args.sbert_model
            )
            
            logger.info("SBERT classification completed")
            evaluate_results(sbert_results)
        except Exception as e:
            logger.error(f"Error in SBERT classification: {e}")
    
    if args.method in ['adaptive', 'both']:
        logger.info("Running adaptive ensemble classification")
        try:
            adaptive_results = run_adaptive_ensemble_classification(
                companies_df,
                taxonomy_df,
                top_k=args.top_k,
                threshold=args.threshold,
                use_sbert=(args.method == 'both'),  # Use SBERT if it was also run
                sbert_model=args.sbert_model,
                train_meta_model=args.train_meta_model,
                meta_model_samples=args.meta_model_samples
            )
            
            logger.info("Adaptive ensemble classification completed")
            evaluate_results(adaptive_results)
        except Exception as e:
            logger.error(f"Error in adaptive ensemble classification: {e}")
    
    logger.info(f"All classification methods completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()