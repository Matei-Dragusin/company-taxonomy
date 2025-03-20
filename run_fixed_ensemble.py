#!/usr/bin/env python
"""
Script to run the fixed optimized ensemble classification for insurance taxonomy.
This script uses the FixedOptimizedEnsembleClassifier which addresses issues
in the original implementation.
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
        logging.FileHandler("fixed_ensemble.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure necessary data directories exist"""
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

def run_fixed_ensemble_classification(companies_df: pd.DataFrame, 
                                     taxonomy_df: pd.DataFrame, 
                                     top_k: int = 5, 
                                     threshold: float = 0.08,
                                     batch_size: int = 100,
                                     tfidf_weight: float = 0.5,
                                     wordnet_weight: float = 0.25,
                                     keyword_weight: float = 0.25,
                                     output_file: str = "fixed_ensemble_classified_companies.csv") -> pd.DataFrame:
    """
    Run fixed optimized ensemble classification on company data.
    
    Args:
        companies_df: DataFrame with preprocessed company data
        taxonomy_df: DataFrame with taxonomy data
        top_k: Maximum number of labels to assign to a company
        threshold: Minimum similarity threshold to assign a label
        batch_size: Number of companies to process in each batch
        tfidf_weight: Weight for TF-IDF similarity in ensemble
        wordnet_weight: Weight for WordNet similarity in ensemble
        keyword_weight: Weight for keyword similarity in ensemble
        output_file: Name of the output file
        
    Returns:
        DataFrame with classified companies
    """
    logger.info("Starting fixed optimized ensemble classification")
    start_time = time.time()
    
    # Import FixedOptimizedEnsembleClassifier
    try:
        from src.ensemble.fixed_optimized_ensemble import FixedOptimizedEnsembleClassifier
    except ImportError as e:
        logger.error(f"Error importing FixedOptimizedEnsembleClassifier: {e}")
        raise
    
    # Initialize ensemble classifier
    logger.info("Initializing FixedOptimizedEnsembleClassifier")
    ensemble_classifier = FixedOptimizedEnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        synonym_cache_size=2048
    )
    
    # Display input data summary
    logger.info(f"Input data summary:")
    logger.info(f"- Companies: {len(companies_df)} rows with {len(companies_df.columns)} columns")
    logger.info(f"- Taxonomy: {len(taxonomy_df)} rows with {len(taxonomy_df.columns)} columns")
    
    # Check for required columns
    if 'combined_features' not in companies_df.columns:
        logger.error("Required column 'combined_features' not found in companies DataFrame")
        raise ValueError("Required column 'combined_features' not found in companies DataFrame")
        
    if 'cleaned_label' not in taxonomy_df.columns:
        logger.error("Required column 'cleaned_label' not found in taxonomy DataFrame")
        raise ValueError("Required column 'cleaned_label' not found in taxonomy DataFrame")
    
    # Classify companies
    try:
        logger.info(f"Starting classification with parameters: top_k={top_k}, threshold={threshold}, batch_size={batch_size}")
        classified_companies = ensemble_classifier.ensemble_classify(
            companies_df,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features',
            batch_size=batch_size
        )
        
        logger.info(f"Classification completed successfully for {len(classified_companies)} companies")
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Save models for future use
    try:
        ensemble_classifier.save_models()
        logger.info("Ensemble models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
    
    # Save results
    try:
        output_path = os.path.join('data/processed', output_file)
        classified_companies.to_csv(output_path, index=False)
        logger.info(f"Classification results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    logger.info(f"Total classification process completed in {time.time() - start_time:.2f} seconds")
    
    return classified_companies

def main():
    """Main function to run the fixed optimized ensemble classification"""
    parser = argparse.ArgumentParser(description='Fixed optimized ensemble classification for insurance taxonomy')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Name of the company file (default: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Name of the taxonomy file (default: insurance_taxonomy.csv)')
    
    parser.add_argument('--top-k', type=int, default=5,
                        help='Maximum number of labels to assign to a company (default: 5)')
    
    parser.add_argument('--threshold', type=float, default=0.08,
                        help='Minimum similarity threshold to assign a label (default: 0.08)')
    
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of companies to process in each batch (default: 100)')
    
    parser.add_argument('--tfidf-weight', type=float, default=0.5,
                        help='Weight for TF-IDF similarity in ensemble (default: 0.5)')
    
    parser.add_argument('--wordnet-weight', type=float, default=0.25,
                        help='Weight for WordNet similarity in ensemble (default: 0.25)')
    
    parser.add_argument('--keyword-weight', type=float, default=0.25,
                        help='Weight for keyword similarity in ensemble (default: 0.25)')
    
    parser.add_argument('--output-file', type=str, default='fixed_ensemble_classified_companies.csv',
                        help='Name of the output file (default: fixed_ensemble_classified_companies.csv)')
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing and use existing processed files')
    
    args = parser.parse_args()
    
    logger.info("Starting fixed optimized ensemble classification for insurance taxonomy")
    total_start_time = time.time()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if input files exist
    raw_data_path = 'data/raw/'
    if not args.skip_preprocessing and not os.path.exists(os.path.join(raw_data_path, args.company_file)):
        logger.error(f"Error: Company file not found at {os.path.join(raw_data_path, args.company_file)}")
        return
    
    if not args.skip_preprocessing and not os.path.exists(os.path.join(raw_data_path, args.taxonomy_file)):
        logger.error(f"Error: Taxonomy file not found at {os.path.join(raw_data_path, args.taxonomy_file)}")
        return
    
    # Load and preprocess data
    if args.skip_preprocessing:
        logger.info("Skipping preprocessing, loading processed data...")
        processed_path = 'data/processed/'
        try:
            companies_df = pd.read_csv(os.path.join(processed_path, 'processed_companies.csv'))
            taxonomy_df = pd.read_csv(os.path.join(processed_path, 'processed_taxonomy.csv'))
            logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels from processed files")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return
    else:
        try:
            companies_df, taxonomy_df = load_and_preprocess_data(args.company_file, args.taxonomy_file)
        except Exception as e:
            logger.error(f"Error in preprocessing stage: {e}")
            return
    
    # Run classification
    try:
        classified_companies = run_fixed_ensemble_classification(
            companies_df,
            taxonomy_df,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            tfidf_weight=args.tfidf_weight,
            wordnet_weight=args.wordnet_weight,
            keyword_weight=args.keyword_weight,
            output_file=args.output_file
        )
        
        # Display statistics about results
        total_companies = len(classified_companies)
        classified_count = sum(1 for labels in classified_companies['insurance_labels'] if labels)
        avg_labels_per_company = sum(len(labels) for labels in classified_companies['insurance_labels']) / total_companies if total_companies > 0 else 0
        
        logger.info(f"Classification complete:")
        logger.info(f"  - Total companies processed: {total_companies}")
        logger.info(f"  - Companies successfully classified: {classified_count} ({classified_count/total_companies*100:.1f}%)")
        logger.info(f"  - Average number of labels per company: {avg_labels_per_company:.2f}")
        logger.info(f"Complete results available at: data/processed/{args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in classification stage: {e}")
        return
    
    total_runtime = time.time() - total_start_time
    logger.info(f"Total script runtime: {total_runtime:.2f} seconds")

if __name__ == "__main__":
    main()