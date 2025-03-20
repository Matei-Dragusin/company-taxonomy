#!/usr/bin/env python
"""
Script to run the ensemble classification for insurance taxonomy.
This script uses the EnsembleClassifier to classify companies.
"""

import os
import sys
import pandas as pd
import logging
import argparse
from typing import Tuple

# Add the project root directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from src.preprocessing.preprocessing import DataPreprocessor
from src.ensemble.ensemble_classifier import EnsembleClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Process data
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file=company_file,
        taxonomy_file=taxonomy_file,
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )
    
    return companies_df, taxonomy_df

def run_ensemble_classification(companies_df: pd.DataFrame, 
                               taxonomy_df: pd.DataFrame, 
                               top_k: int = 5, 
                               threshold: float = 0.08,
                               tfidf_weight: float = 0.5,
                               wordnet_weight: float = 0.3,
                               keyword_weight: float = 0.2,
                               output_file: str = "ensemble_classified_companies.csv") -> pd.DataFrame:
    """
    Run ensemble classification on company data.
    
    Args:
        companies_df: DataFrame with preprocessed company data
        taxonomy_df: DataFrame with taxonomy data
        top_k: Maximum number of labels to assign to a company
        threshold: Minimum similarity threshold to assign a label
        tfidf_weight: Weight for TF-IDF similarity in ensemble
        wordnet_weight: Weight for WordNet similarity in ensemble
        keyword_weight: Weight for keyword similarity in ensemble
        output_file: Name of the output file
        
    Returns:
        DataFrame with classified companies
    """
    logger.info("Starting ensemble classification")
    
    # Initialize ensemble classifier
    ensemble_classifier = EnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight
    )
    
    # Classify companies
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=top_k,
        threshold=threshold,
        company_text_column='combined_features'
    )
    
    # Save models for future use
    ensemble_classifier.save_models()
    
    # Save results
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"Classification results saved to {output_path}")
    
    return classified_companies

def main():
    """Main function to run the ensemble classification"""
    parser = argparse.ArgumentParser(description='Ensemble classification for insurance taxonomy')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Name of the company file (default: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Name of the taxonomy file (default: insurance_taxonomy.csv)')
    
    parser.add_argument('--top-k', type=int, default=5,
                        help='Maximum number of labels to assign to a company (default: 5)')
    
    parser.add_argument('--threshold', type=float, default=0.08,
                        help='Minimum similarity threshold to assign a label (default: 0.08)')
    
    parser.add_argument('--tfidf-weight', type=float, default=0.5,
                        help='Weight for TF-IDF similarity in ensemble (default: 0.5)')
    
    parser.add_argument('--wordnet-weight', type=float, default=0.3,
                        help='Weight for WordNet similarity in ensemble (default: 0.3)')
    
    parser.add_argument('--keyword-weight', type=float, default=0.2,
                        help='Weight for keyword similarity in ensemble (default: 0.2)')
    
    parser.add_argument('--output-file', type=str, default='ensemble_classified_companies.csv',
                        help='Name of the output file (default: ensemble_classified_companies.csv)')
    
    args = parser.parse_args()
    
    logger.info("Starting ensemble classification for insurance taxonomy")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if input files exist
    raw_data_path = 'data/raw/'
    if not os.path.exists(os.path.join(raw_data_path, args.company_file)):
        logger.error(f"Error: Company file not found at {os.path.join(raw_data_path, args.company_file)}")
        return
    
    if not os.path.exists(os.path.join(raw_data_path, args.taxonomy_file)):
        logger.error(f"Error: Taxonomy file not found at {os.path.join(raw_data_path, args.taxonomy_file)}")
        return
    
    # Load and preprocess data
    companies_df, taxonomy_df = load_and_preprocess_data(args.company_file, args.taxonomy_file)
    
    # Run classification
    classified_companies = run_ensemble_classification(
        companies_df,
        taxonomy_df,
        top_k=args.top_k,
        threshold=args.threshold,
        tfidf_weight=args.tfidf_weight,
        wordnet_weight=args.wordnet_weight,
        keyword_weight=args.keyword_weight,
        output_file=args.output_file
    )
    
    # Display statistics about results
    total_companies = len(classified_companies)
    classified_count = sum(1 for labels in classified_companies['insurance_labels'] if labels)
    avg_labels_per_company = sum(len(labels) for labels in classified_companies['insurance_labels']) / total_companies
    
    logger.info(f"Classification complete:")
    logger.info(f"  - Total companies processed: {total_companies}")
    logger.info(f"  - Companies successfully classified: {classified_count} ({classified_count/total_companies*100:.1f}%)")
    logger.info(f"  - Average number of labels per company: {avg_labels_per_company:.2f}")
    logger.info(f"Complete results available at: data/processed/{args.output_file}")

if __name__ == "__main__":
    main()  