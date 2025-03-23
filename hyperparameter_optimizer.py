#!/usr/bin/env python
"""
Script for automatic hyperparameter optimization of the classifier.
Uses Optuna to find the optimal values for the classifier parameters.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
import time
import optuna
from typing import Dict, List, Tuple, Optional
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add the project's root directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.preprocessing.preprocessing import DataPreprocessor
from src.feature_engineering.tfidf_processor import TFIDFProcessor
from src.ensemble.ensemble_classifier import EnsembleClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'parameter_optimization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
def ensure_directories_exist():
    """Ensure necessary directories exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'results', 'logs', 'optimization_results']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

# Load and preprocess data
def load_and_preprocess_data(company_file: str, taxonomy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess company and taxonomy data.
    
    Args:
        company_file: Name of the company file
        taxonomy_file: Name of the taxonomy file
        
    Returns:
        Tuple with preprocessed DataFrames for companies and taxonomy
    """
    logger.info(f"Loading and preprocessing data from {company_file} and {taxonomy_file}")
    
    # Check if preprocessed data exists
    processed_companies_path = os.path.join('data/processed', 'processed_companies.csv')
    processed_taxonomy_path = os.path.join('data/processed', 'processed_taxonomy.csv')
    
    if os.path.exists(processed_companies_path) and os.path.exists(processed_taxonomy_path):
        logger.info("Loading existing preprocessed data...")
        companies_df = pd.read_csv(processed_companies_path)
        taxonomy_df = pd.read_csv(processed_taxonomy_path)
        return companies_df, taxonomy_df
    
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Process the data
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file=company_file,
        taxonomy_file=taxonomy_file,
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )
    
    return companies_df, taxonomy_df

# Function to evaluate a model with a specific set of parameters
def evaluate_parameters(
    companies_df: pd.DataFrame, 
    taxonomy_df: pd.DataFrame,
    validation_set: pd.DataFrame,
    params: Dict
) -> float:
    """
    Evaluate a set of parameters on the validation data.
    
    Args:
        companies_df: DataFrame with company data
        taxonomy_df: DataFrame with taxonomy
        validation_set: DataFrame with annotated validation set
        params: Dictionary with parameters to test
        
    Returns:
        Performance score (higher is better)
    """
    # Extract parameters from the dictionary
    top_k = params.get('top_k', 5)
    threshold = params.get('threshold', 0.08)
    batch_size = params.get('batch_size', 100)
    tfidf_weight = params.get('tfidf_weight', 0.5)
    wordnet_weight = params.get('wordnet_weight', 0.25)
    keyword_weight = params.get('keyword_weight', 0.25)
    
    # Create the classifier with the specified parameters
    ensemble_classifier = EnsembleClassifier(
        models_path='models/temp/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        optimizer_mode=True
    )
    
    # Run classification on a small subset for efficiency
    sample_size = min(200, len(companies_df))
    sample_companies = companies_df.sample(n=sample_size, random_state=42)
    
    try:
        # Classify the companies
        classified_companies = ensemble_classifier.ensemble_classify(
            sample_companies,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features',
            batch_size=batch_size
        )
        
        # Calculate performance metrics
        # If we have manually annotated data, we can calculate precision, recall, F1
        # Otherwise, we can use heuristic metrics:
        
        # 1. Coverage (percentage of companies with at least one label)
        coverage = (classified_companies['insurance_label'] != 'Unclassified').mean()
        
        # 2. Average confidence score
        confidence_scores = []
        for scores in classified_companies['insurance_label_scores']:
            if isinstance(scores, list) and len(scores) > 0:
                confidence_scores.append(np.mean(scores))
            elif isinstance(scores, str) and scores.startswith('[') and scores.endswith(']'):
                try:
                    parsed_scores = eval(scores)
                    if len(parsed_scores) > 0:
                        confidence_scores.append(np.mean(parsed_scores))
                except:
                    pass
        
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # 3. Label diversity (number of unique labels used)
        all_labels = []
        for labels in classified_companies['insurance_labels']:
            if isinstance(labels, list):
                all_labels.extend(labels)
            elif isinstance(labels, str) and labels.startswith('[') and labels.endswith(']'):
                try:
                    parsed_labels = eval(labels)
                    all_labels.extend(parsed_labels)
                except:
                    pass
        
        unique_labels = len(set(all_labels))
        label_diversity = unique_labels / len(taxonomy_df) if len(taxonomy_df) > 0 else 0
        
        # Combine metrics into a final score
        # We can adjust the weights based on the importance of each metric
        final_score = (0.4 * coverage) + (0.4 * mean_confidence) + (0.2 * label_diversity)
        
        # If we have manually annotated validation data, use F1 score
        if validation_set is not None and len(validation_set) > 0:
            # Implement evaluation logic on annotated data
            # ...
            pass
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error evaluating parameters: {e}")
        return 0.0  # Return a minimum score in case of error

# Objective function for Optuna
def objective(trial, companies_df, taxonomy_df, validation_set=None):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        companies_df: DataFrame with company data
        taxonomy_df: DataFrame with taxonomy
        validation_set: Optional validation data
        
    Returns:
        Performance score
    """
    # Define the parameter search space
    params = {
        'top_k': trial.suggest_int('top_k', 1, 10),
        'threshold': trial.suggest_float('threshold', 0.01, 0.3),
        'batch_size': trial.suggest_categorical('batch_size', [50, 100, 200]),
    }
    
    # Use a different approach to ensure weights sum to 1
    # Suggest relative weights, then normalize them
    tfidf_relative = trial.suggest_float('tfidf_relative', 0.1, 1.0)
    wordnet_relative = trial.suggest_float('wordnet_relative', 0.1, 1.0)
    keyword_relative = trial.suggest_float('keyword_relative', 0.1, 1.0)
    
    # Normalize weights to sum to 1
    total_weight = tfidf_relative + wordnet_relative + keyword_relative
    params['tfidf_weight'] = tfidf_relative / total_weight
    params['wordnet_weight'] = wordnet_relative / total_weight
    params['keyword_weight'] = keyword_relative / total_weight
    
    # Evaluate the parameters
    score = evaluate_parameters(companies_df, taxonomy_df, validation_set, params)
    
    return score

# Function for hyperparameter optimization
def optimize_hyperparameters(
    companies_df: pd.DataFrame, 
    taxonomy_df: pd.DataFrame,
    validation_set: Optional[pd.DataFrame] = None,
    n_trials: int = 50,
    timeout: int = 3600,  # Default 1 hour
    study_name: str = "insurance_taxonomy_optimization"
) -> Dict:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        companies_df: DataFrame with company data
        taxonomy_df: DataFrame with taxonomy
        validation_set: Optional validation data
        n_trials: Number of trials for optimization
        timeout: Maximum time (in seconds) for optimization
        study_name: Name of the Optuna study
        
    Returns:
        Dictionary with optimized parameters
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    # Create a directory to store the study data
    study_dir = os.path.join('optimization_results')
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    
    # Create a new Optuna study
    db_path = f"sqlite:///{os.path.join(study_dir, f'{study_name}.db')}"
    
    # Set pruner and sampler for efficiency
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = TPESampler(seed=42)
    
    # Create or load the study
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )
    
    # Run the optimization
    start_time = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, companies_df, taxonomy_df, validation_set),
            n_trials=n_trials,
            timeout=timeout
        )
    except KeyboardInterrupt:
        logger.info("Optimization manually interrupted.")
    
    duration = time.time() - start_time
    logger.info(f"Optimization completed in {duration:.2f} seconds.")
    
    # Get the optimized parameters
    best_params = study.best_params
    best_score = study.best_value
    
    # Display results
    logger.info(f"Best score: {best_score}")
    logger.info(f"Optimized parameters: {best_params}")
    
    # Save results to a file
    results_file = os.path.join(study_dir, f"{study_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Best score: {best_score}\n")
        f.write(f"Optimized parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\nTrial history:\n")
        # Sort only trials with a valid value
        valid_trials = [t for t in study.trials if t.value is not None]
        for trial in sorted(valid_trials, key=lambda t: t.value, reverse=True):
            f.write(f"  Trial {trial.number}, Score: {trial.value}\n")
            f.write(f"  Params: {trial.params}\n\n")
    
    # Save plots for hyperparameter analysis
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_param_importances, plot_optimization_history
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(study_dir, f"{study_name}_history.png"))
        
        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(study_dir, f"{study_name}_importance.png"))
        
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return best_params

# Function to run classification with optimized parameters
def run_with_optimized_parameters(
    companies_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    optimized_params: Dict,
    output_file: str = "optimized_classification.csv"
) -> pd.DataFrame:
    """
    Run classification with optimized parameters.
    
    Args:
        companies_df: DataFrame with company data
        taxonomy_df: DataFrame with taxonomy
        optimized_params: Dictionary with optimized parameters
        output_file: Name of the output file
        
    Returns:
        DataFrame with classification results
    """
    logger.info(f"Running classification with optimized parameters: {optimized_params}")
    
    # Extract parameters
    top_k = optimized_params.get('top_k', 5)
    threshold = optimized_params.get('threshold', 0.08)
    batch_size = optimized_params.get('batch_size', 100)
    tfidf_weight = optimized_params.get('tfidf_weight', 0.5)
    wordnet_weight = optimized_params.get('wordnet_weight', 0.25)
    keyword_weight = optimized_params.get('keyword_weight', 0.25)
    
    # Initialize the ensemble classifier
    ensemble_classifier = EnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        optimizer_mode=True
    )
    
    # Classify the companies
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=top_k,
        threshold=threshold,
        company_text_column='combined_features',
        batch_size=batch_size
    )
    
    # Save the models for future use
    ensemble_classifier.save_models(filename_prefix='optimized_ensemble_model')
    
    # Save the results
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Save a simplified file
    ensemble_classifier.export_description_label_csv(
        classified_companies,
        output_path=os.path.join('data/processed', 'optimized_description_label.csv'),
        description_column='description'
    )
    
    return classified_companies

# Main function
def main():
    """Main function to run hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for taxonomy classification')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Name of the company file (default: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Name of the taxonomy file (default: insurance_taxonomy.csv)')
    
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for optimization (default: 50)')
    
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time in seconds for optimization (default: 3600)')
    
    parser.add_argument('--study-name', type=str, default='insurance_taxonomy_optimization',
                        help='Name of the Optuna study (default: insurance_taxonomy_optimization)')
    
    parser.add_argument('--run-optimized', action='store_true',
                        help='Run classification with optimized parameters after optimization')
    
    parser.add_argument('--output-file', type=str, default='optimized_classified_companies.csv',
                        help='Name of the output file for classification (default: optimized_classified_companies.csv)')
    
    args = parser.parse_args()
    
    logger.info("Starting hyperparameter optimization for taxonomy classification")
    
    # Ensure necessary directories exist
    ensure_directories_exist()
    
    # Load and preprocess data
    try:
        companies_df, taxonomy_df = load_and_preprocess_data(
            args.company_file, 
            args.taxonomy_file
        )
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {e}")
        return
    
    # Optimize hyperparameters
    try:
        optimized_params = optimize_hyperparameters(
            companies_df,
            taxonomy_df,
            validation_set=None,  # Add manually annotated data here if available
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=args.study_name
        )
        
        logger.info(f"Optimized parameters obtained: {optimized_params}")
        
        # Run classification with optimized parameters if requested
        if args.run_optimized:
            classified_companies = run_with_optimized_parameters(
                companies_df,
                taxonomy_df,
                optimized_params,
                output_file=args.output_file
            )
            
            logger.info(f"Classification with optimized parameters completed!")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    logger.info("Hyperparameter optimization completed!")

if __name__ == "__main__":
    main()
