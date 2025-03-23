#!/usr/bin/env python
"""
Unified script for classifying companies into the insurance taxonomy.
Supports both basic classification and optimized classification for large datasets.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import time
from typing import Dict, List, Tuple

# Add the project root directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from src.preprocessing.preprocessing import DataPreprocessor
from src.feature_engineering.tfidf_processor import TFIDFProcessor
from src.ensemble.ensemble_classifier import EnsembleClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure necessary directories exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def load_and_preprocess_data(company_file: str, taxonomy_file: str, skip_preprocessing: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess company and taxonomy data.
    
    Args:
        company_file: Name of the company file
        taxonomy_file: Name of the taxonomy file
        skip_preprocessing: Whether to load directly from processed files
        
    Returns:
        Tuple with preprocessed company and taxonomy DataFrames
    """
    if skip_preprocessing:
        logger.info("Loading preprocessed data...")
        processed_path = 'data/processed/'
        companies_df = pd.read_csv(os.path.join(processed_path, 'processed_companies.csv'))
        taxonomy_df = pd.read_csv(os.path.join(processed_path, 'processed_taxonomy.csv'))
        logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        return companies_df, taxonomy_df
    
    logger.info(f"Loading and preprocessing data from {company_file} and {taxonomy_file}")
    start_time = time.time()
    
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
    
    logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    return companies_df, taxonomy_df

def run_classification(companies_df: pd.DataFrame, 
                    taxonomy_df: pd.DataFrame,
                    use_optimizer: bool = True,
                    top_k: int = 3,  # Reduced from 5 to improve precision
                    threshold: float = 0.05,  # Lower threshold to increase matches
                    batch_size: int = 100,
                    tfidf_weight: float = 0.6,  # Increased TF-IDF weight for better text matching
                    wordnet_weight: float = 0.25,
                    keyword_weight: float = 0.15,  # Reduced slightly to prioritize TF-IDF
                    ensure_one_tag: bool = True,  # New parameter to ensure every company has a tag
                    output_file: str = "classified_companies.csv",
                    description_label_file: str = "description_label_results.csv",
                    description_column: str = "description",
                    include_scores: bool = True) -> pd.DataFrame:
    """
    Run classification on company data.
    
    Args:
        companies_df: DataFrame with preprocessed company data
        taxonomy_df: DataFrame with taxonomy data
        use_optimizer: Whether to use optimized mode
        top_k: Maximum number of labels to assign to a company
        threshold: Minimum similarity threshold to assign a label
        batch_size: Number of companies to process in each batch
        tfidf_weight: Weight for TF-IDF similarity in ensemble
        wordnet_weight: Weight for WordNet similarity in ensemble
        keyword_weight: Weight for keyword similarity in ensemble
        ensure_one_tag: Whether to ensure each company has at least one tag
        output_file: Name of the output file
        description_label_file: Name of the simplified description-label output file
        description_column: Column containing company descriptions
        include_scores: Whether to include confidence scores in the output
        
    Returns:
        DataFrame with classified companies
    """
    logger.info(f"Starting classification using {'optimized' if use_optimizer else 'basic'} mode")
    start_time = time.time()
    
    # Initialize ensemble classifier
    ensemble_classifier = EnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        optimizer_mode=use_optimizer,
        synonym_cache_size=2048
    )
    
    # Classify companies
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=top_k,
        threshold=threshold,
        company_text_column='combined_features',
        batch_size=batch_size,
        ensure_one_tag=ensure_one_tag
    )
    
    # Save models for future use
    ensemble_classifier.save_models()
    
    # Save full results
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"Classification results saved to {output_path}")
    
    # Save simplified description-label results
    if description_label_file:
        try:
            dl_output_path = os.path.join('data/processed', description_label_file)
            ensemble_classifier.export_description_label_csv(
                classified_companies,
                output_path=dl_output_path,
                description_column=description_column,
                include_scores=include_scores
            )
            logger.info(f"Description-label results saved to {dl_output_path}")
        except Exception as e:
            logger.error(f"Error exporting description-label results: {e}")
    
    logger.info(f"Total classification process completed in {time.time() - start_time:.2f} seconds")
    
    return classified_companies

def evaluate_results(classified_df: pd.DataFrame, output_dir: str = 'results/') -> Dict:
    """
    Evaluate classification results and generate statistics.
    
    Args:
        classified_df: DataFrame with classification results
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation statistics
    """
    logger.info("Evaluating classification results")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Basic statistics
    total_companies = len(classified_df)
    companies_with_labels = 0
    total_labels = 0
    label_counts = {}
    
    # Distribution of labels per company
    label_distribution = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}
    
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
        
        # Count number of labels per company
        num_labels = len(labels) if isinstance(labels, list) else 0
        
        if num_labels >= 4:
            label_distribution['4+'] += 1
        else:
            label_distribution[num_labels] += 1
        
        # Update counters
        if num_labels > 0:
            companies_with_labels += 1
            total_labels += num_labels
            
            # Count individual labels
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate additional metrics
    coverage = (companies_with_labels / total_companies) * 100 if total_companies > 0 else 0
    avg_labels_per_company = total_labels / total_companies if total_companies > 0 else 0
    
    # Compile results
    eval_results = {
        'total_companies': total_companies,
        'companies_with_labels': companies_with_labels,
        'coverage_percentage': coverage,
        'total_labels_assigned': total_labels,
        'avg_labels_per_company': avg_labels_per_company,
        'unique_labels_used': len(label_counts),
        'label_distribution': label_distribution,
        'top_labels': sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    # Display results
    print("\n=== CLASSIFICATION EVALUATION ===")
    print(f"Total companies: {total_companies}")
    print(f"Companies with labels: {companies_with_labels} ({coverage:.2f}%)")
    print(f"Average labels per company: {avg_labels_per_company:.2f}")
    print(f"Unique labels used: {len(label_counts)}")
    
    print("\n== DISTRIBUTION OF LABELS PER COMPANY ==")
    for label_count, num_companies in label_distribution.items():
        print(f"{label_count} labels: {num_companies} companies")
    
    print("\n== TOP 10 MOST COMMON LABELS ==")
    for label, count in eval_results['top_labels']:
        print(f"{label}: {count}")
    
    # Generate plots
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot label distribution
        labels = list(label_distribution.keys())
        values = [label_distribution[k] for k in labels]
        
        plt.bar([str(x) for x in labels], values)  # Convert labels to strings to avoid type issues
        plt.xlabel('Number of Labels')
        plt.ylabel('Number of Companies')
        plt.title('Distribution of Labels per Company')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
        logger.info(f"Saved label distribution plot to {output_dir}")
    except Exception as e:
        logger.warning(f"Error generating plots: {e}")
    
    return eval_results

def load_optimized_parameters():
    """Încarcă parametrii optimizați din fișierul rezultat"""
    opt_file = 'optimization_results/insurance_taxonomy_optimization_results.txt'
    
    if not os.path.exists(opt_file):
        logger.warning("Fișierul cu parametri optimizați nu există. Se folosesc valorile implicite.")
        return None
        
    params = {}
    try:
        with open(opt_file, 'r') as f:
            lines = f.readlines()
            param_section = False
            
            for line in lines:
                if line.startswith("Parametri optimizați:"):
                    param_section = True
                    continue
                
                if param_section and line.strip() and ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convertim valorile la tipul potrivit
                    if key in ['top_k', 'batch_size']:
                        params[key] = int(value)
                    else:
                        params[key] = float(value)
        
        return params
    except Exception as e:
        logger.warning(f"Eroare la încărcarea parametrilor optimizați: {e}")
        return None

def main():
    """Main function to run the classification"""
    parser = argparse.ArgumentParser(description='Unified classification for insurance taxonomy')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Name of the company file (default: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Name of the taxonomy file (default: insurance_taxonomy.csv)')
    
    parser.add_argument('--use-optimizer', action='store_true', default=True,
                        help='Use optimized mode for large datasets (default: True)')
    
    parser.add_argument('--top-k', type=int, default=3,
                        help='Maximum number of labels to assign to a company (default: 3)')
    
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Minimum similarity threshold to assign a label (default: 0.05)')
    
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of companies to process in each batch (default: 100)')
    
    parser.add_argument('--tfidf-weight', type=float, default=0.6,
                        help='Weight for TF-IDF similarity in ensemble (default: 0.6)')
    
    parser.add_argument('--wordnet-weight', type=float, default=0.25,
                        help='Weight for WordNet similarity in ensemble (default: 0.25)')
    
    parser.add_argument('--keyword-weight', type=float, default=0.15,
                        help='Weight for keyword similarity in ensemble (default: 0.15)')
    
    parser.add_argument('--ensure-one-tag', action='store_true', default=True,
                        help='Ensure each company has at least one tag (default: True)')
    
    parser.add_argument('--output-file', type=str, default='classified_companies.csv',
                        help='Name of the output file (default: classified_companies.csv)')
    
    parser.add_argument('--description-label-file', type=str, default='description_label_results.csv',
                        help='Name of the simplified description-label output file (default: description_label_results.csv)')
    
    parser.add_argument('--description-column', type=str, default='description',
                        help='Column containing company descriptions (default: description)')
    
    parser.add_argument('--include-scores', action='store_true', default=True,
                        help='Include confidence scores in the output (default: True)')
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing and use existing processed files')
    
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate classification results after completion')
    
    parser.add_argument('--use-optimized-params', action='store_true',
                        help='Use optimized parameters from hyperparameter search')
    
    args = parser.parse_args()
    
    logger.info("Starting unified classification for insurance taxonomy")
    total_start_time = time.time()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if input files exist
    if not args.skip_preprocessing:
        raw_data_path = 'data/raw/'
        if not os.path.exists(os.path.join(raw_data_path, args.company_file)):
            logger.error(f"Error: Company file not found at {os.path.join(raw_data_path, args.company_file)}")
            return
        
        if not os.path.exists(os.path.join(raw_data_path, args.taxonomy_file)):
            logger.error(f"Error: Taxonomy file not found at {os.path.join(raw_data_path, args.taxonomy_file)}")
            return
    
    # Load and preprocess data
    try:
        companies_df, taxonomy_df = load_and_preprocess_data(
            args.company_file, 
            args.taxonomy_file,
            args.skip_preprocessing
        )
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Load optimized parameters if requested
    if args.use_optimized_params:
        opt_params = load_optimized_parameters()
        if opt_params:
            logger.info(f"Using optimized parameters: {opt_params}")
            # Update args with optimized parameters
            for param, value in opt_params.items():
                if hasattr(args, param.replace('-', '_')):
                    setattr(args, param.replace('-', '_'), value)
    
    # Run classification
    try:
        classified_companies = run_classification(
            companies_df,
            taxonomy_df,
            use_optimizer=args.use_optimizer,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            tfidf_weight=args.tfidf_weight,
            wordnet_weight=args.wordnet_weight,
            keyword_weight=args.keyword_weight,
            ensure_one_tag=args.ensure_one_tag,
            output_file=args.output_file,
            description_label_file=args.description_label_file,
            description_column=args.description_column,
            include_scores=args.include_scores
        )
        
        # Evaluate results if requested
        if args.evaluate:
            evaluate_results(classified_companies)
            
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    total_runtime = time.time() - total_start_time
    logger.info(f"Total script runtime: {total_runtime:.2f} seconds")
    
if __name__ == "__main__":
    main()