#!/usr/bin/env python
"""
Script for optimizing classification thresholds for companies in the insurance taxonomy.
This script runs the classifier with various thresholds and weight configurations
to find the optimal parameters that produce the best results.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add the project root directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("threshold_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Ensure necessary directories exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'results', 'results/threshold_tests']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
            
def load_data(preprocessed_companies_path: str, preprocessed_taxonomy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed company and taxonomy data.
    
    Args:
        preprocessed_companies_path: Path to preprocessed companies CSV
        preprocessed_taxonomy_path: Path to preprocessed taxonomy CSV
        
    Returns:
        Tuple of DataFrames for companies and taxonomy
    """
    logger.info(f"Loading preprocessed data from {preprocessed_companies_path} and {preprocessed_taxonomy_path}")
    
    try:
        companies_df = pd.read_csv(preprocessed_companies_path)
        taxonomy_df = pd.read_csv(preprocessed_taxonomy_path)
        
        logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        return companies_df, taxonomy_df
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        raise

def run_classification(companies_df: pd.DataFrame, 
                     taxonomy_df: pd.DataFrame,
                     threshold: float,
                     top_k: int,
                     tfidf_weight: float,
                     wordnet_weight: float,
                     keyword_weight: float) -> pd.DataFrame:
    """
    Run classification with specified parameters.
    
    Args:
        companies_df: Preprocessed companies DataFrame
        taxonomy_df: Preprocessed taxonomy DataFrame
        threshold: Similarity threshold to use
        top_k: Maximum number of labels to assign
        tfidf_weight: Weight for TF-IDF similarity
        wordnet_weight: Weight for WordNet similarity
        keyword_weight: Weight for keyword similarity
        
    Returns:
        DataFrame with classification results
    """
    
    logger.info(f"Running classification with: threshold={threshold}, top_k={top_k}, "
                f"weights=(tfidf={tfidf_weight}, wordnet={wordnet_weight}, keyword={keyword_weight})")
    
    # Import the classification module
    try:
        from src.ensemble.fixed_optimized_ensemble import FixedOptimizedEnsembleClassifier
    except ImportError:
        try:
            from src.ensemble.optimized_ensemble_classifier import OptimizedEnsembleClassifier as FixedOptimizedEnsembleClassifier
            logger.info("Using OptimizedEnsembleClassifier as FixedOptimizedEnsembleClassifier")
        except ImportError:
            from src.ensemble.ensemble_classifier import EnsembleClassifier as FixedOptimizedEnsembleClassifier
            logger.warning("Falling back to regular EnsembleClassifier")
    
    # Create classifier with specified weights
    classifier = FixedOptimizedEnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight
    )
    
    # Run classification
    try:
        classified_df = classifier.ensemble_classify(
            companies_df,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features'
        )
        
        return classified_df
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return empty DataFrame with same structure in case of error
        return companies_df.copy()

def evaluate_classification(classified_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate classification quality using various metrics.
    
    Args:
        classified_df: DataFrame with classification results
        
    Returns:
        Dictionary of evaluation metrics
    """
    
    # Basic validation - check if required columns exist
    required_columns = ['insurance_labels', 'insurance_label_scores']
    for col in required_columns:
        if col not in classified_df.columns:
            logger.error(f"Column '{col}' not found in classification results")
            return {}
    
    total_companies = len(classified_df)
    
    # Count companies with labels
    companies_with_labels = 0
    total_labels = 0
    total_score = 0
    label_counts = defaultdict(int)
    
    # Distribution of labels per company
    label_distribution = defaultdict(int)
    
    for _, row in classified_df.iterrows():
        labels = row['insurance_labels']
        scores = row['insurance_label_scores']
        
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
        label_distribution[num_labels] += 1
        
        # Update counters
        if num_labels > 0:
            companies_with_labels += 1
            total_labels += num_labels
            
            # Count individual labels
            for label in labels:
                label_counts[label] += 1
            
            # Calculate average score
            if isinstance(scores, list) and len(scores) > 0:
                total_score += sum(scores)
    
    # Calculate additional metrics
    coverage = (companies_with_labels / total_companies) * 100 if total_companies > 0 else 0
    avg_labels_per_company = total_labels / total_companies if total_companies > 0 else 0
    avg_labels_per_labeled_company = total_labels / companies_with_labels if companies_with_labels > 0 else 0
    avg_score = total_score / total_labels if total_labels > 0 else 0
    
    # Calculate diversity - how many unique labels are used
    unique_labels = len(label_counts)
    
    # Calculate diversity ratio (unique labels used / total available labels)
    diversity_ratio = unique_labels / len(taxonomy_df) if 'taxonomy_df' in globals() else 0
    
    # Find the 10 most common labels
    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Label balance - are some labels assigned much more than others?
    if len(label_counts) > 0:
        label_values = np.array(list(label_counts.values()))
        label_std = np.std(label_values)
        label_mean = np.mean(label_values)
        label_balance = label_std / label_mean if label_mean > 0 else 0
    else:
        label_balance = 0
    
    # Compile results
    eval_results = {
        'total_companies': total_companies,
        'companies_with_labels': companies_with_labels,
        'coverage_percentage': coverage,
        'total_labels_assigned': total_labels,
        'avg_labels_per_company': avg_labels_per_company,
        'avg_labels_per_labeled_company': avg_labels_per_labeled_company,
        'avg_score': avg_score,
        'unique_labels_used': unique_labels,
        'diversity_ratio': diversity_ratio,
        'label_balance': label_balance,
        'label_distribution': dict(label_distribution),
        'top_labels': top_labels
    }
    
    return eval_results

def calculate_objective_function(eval_results: Dict[str, Any], weight_coverage: float = 0.5, 
                               weight_avg_labels: float = 0.2, weight_avg_score: float = 0.2, 
                               weight_diversity: float = 0.1) -> float:
    """
    Calculate a single objective function score for comparing different parameter sets.
    
    Args:
        eval_results: Evaluation results dictionary
        weight_coverage: Weight for coverage percentage
        weight_avg_labels: Weight for average labels per company
        weight_avg_score: Weight for average score
        weight_diversity: Weight for diversity ratio
        
    Returns:
        Score value (higher is better)
    """
    
    # Extract metrics
    coverage = eval_results.get('coverage_percentage', 0)
    avg_labels = eval_results.get('avg_labels_per_labeled_company', 0)
    avg_score = eval_results.get('avg_score', 0)
    diversity = eval_results.get('diversity_ratio', 0)
    
    # Normalize average labels (optimal is around 2-3 labels per company)
    # We'll use a bell curve that peaks at 2.5 labels
    normalized_avg_labels = np.exp(-((avg_labels - 2.5) ** 2) / 2)
    
    # Calculate weighted score
    score = (
        weight_coverage * coverage / 100 +  # Normalize to 0-1 scale
        weight_avg_labels * normalized_avg_labels +
        weight_avg_score * avg_score +
        weight_diversity * diversity
    )
    
    return score

def save_results(classified_df: pd.DataFrame, eval_results: Dict[str, Any], 
                parameters: Dict[str, Any], score: float, output_dir: str) -> str:
    """
    Save classification results and evaluation metrics.
    
    Args:
        classified_df: DataFrame with classification results
        eval_results: Evaluation results dictionary
        parameters: Parameters used for this run
        score: Objective function score
        output_dir: Directory to save results
        
    Returns:
        Path to saved results directory
    """
    
    # Create timestamp for unique folder name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    threshold = parameters.get('threshold', 0)
    
    # Create directory for this run
    run_dir = os.path.join(output_dir, f"run_{timestamp}_thresh_{threshold:.3f}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save parameters and evaluation metrics
    params_and_results = {**parameters, **eval_results, 'score': score}
    with open(os.path.join(run_dir, "results.txt"), 'w') as f:
        f.write("=== PARAMETERS AND EVALUATION METRICS ===\n\n")
        
        # Parameters section
        f.write("== PARAMETERS ==\n")
        for param, value in parameters.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        # Results section
        f.write("== EVALUATION METRICS ==\n")
        for metric, value in eval_results.items():
            if metric not in ('label_distribution', 'top_labels'):
                f.write(f"{metric}: {value}\n")
        
        # Top labels
        f.write("\n== TOP 10 MOST COMMON LABELS ==\n")
        for label, count in eval_results.get('top_labels', []):
            f.write(f"{label}: {count}\n")
            
        # Label distribution
        f.write("\n== LABEL DISTRIBUTION ==\n")
        for num_labels, count in sorted(eval_results.get('label_distribution', {}).items()):
            f.write(f"{num_labels} labels: {count} companies\n")
            
        # Overall score
        f.write(f"\n== OVERALL SCORE ==\n")
        f.write(f"score: {score}\n")
    
    # Save classified data
    classified_df.to_csv(os.path.join(run_dir, "classified_companies.csv"), index=False)
    
    # Generate and save visualizations
    create_visualizations(eval_results, run_dir)
    
    return run_dir

def create_visualizations(eval_results: Dict[str, Any], output_dir: str) -> None:
    """
    Create and save visualizations of classification results.
    
    Args:
        eval_results: Evaluation results dictionary
        output_dir: Directory to save visualizations
    """
    
    # Create label distribution chart
    plt.figure(figsize=(10, 6))
    label_dist = eval_results.get('label_distribution', {})
    
    if label_dist:
        # Sort by number of labels
        labels = sorted(label_dist.keys())
        values = [label_dist[k] for k in labels]
        
        plt.bar(labels, values)
        plt.xlabel('Number of Labels')
        plt.ylabel('Number of Companies')
        plt.title('Distribution of Labels per Company')
        plt.xticks(labels)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
        plt.close()
    
    # Create top labels chart
    plt.figure(figsize=(12, 8))
    top_labels = eval_results.get('top_labels', [])
    
    if top_labels:
        labels = [label for label, count in top_labels]
        counts = [count for label, count in top_labels]
        
        # Create horizontal bar chart for better label readability
        plt.barh(range(len(labels)), counts, align='center')
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Number of Assignments')
        plt.title('Top 10 Most Common Labels')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_labels.png'))
        plt.close()

def test_multiple_thresholds(companies_df: pd.DataFrame, taxonomy_df: pd.DataFrame, 
                           thresholds: List[float], top_k_values: List[int], 
                           weight_configs: List[Dict[str, float]],
                           objective_weights: Dict[str, float],
                           output_dir: str = 'results/threshold_tests') -> Dict[str, Any]:
    """
    Test multiple threshold and parameter configurations to find the best performing set.
    
    Args:
        companies_df: Preprocessed companies DataFrame
        taxonomy_df: Preprocessed taxonomy DataFrame
        thresholds: List of threshold values to test
        top_k_values: List of top_k values to test
        weight_configs: List of weight configurations to test
        objective_weights: Weights for the objective function
        output_dir: Directory to save results
        
    Returns:
        Dictionary with the best configuration and its results
    """
    
    logger.info("Starting threshold optimization tests")
    
    # Store results from all runs
    all_results = []
    
    # Best configuration
    best_score = -1
    best_config = None
    best_results = None
    best_classified_df = None
    
    # Count total number of combinations to test
    total_tests = len(thresholds) * len(top_k_values) * len(weight_configs)
    logger.info(f"Testing {total_tests} parameter combinations")
    
    # Track progress
    test_count = 0
    
    # Test each combination
    for threshold in thresholds:
        for top_k in top_k_values:
            for weights in weight_configs:
                test_count += 1
                logger.info(f"Running test {test_count}/{total_tests}: "
                           f"threshold={threshold}, top_k={top_k}, weights={weights}")
                
                # Extract individual weights
                tfidf_weight = weights.get('tfidf', 0.5)
                wordnet_weight = weights.get('wordnet', 0.3)
                keyword_weight = weights.get('keyword', 0.2)
                
                # Run classification
                classified_df = run_classification(
                    companies_df,
                    taxonomy_df,
                    threshold=threshold,
                    top_k=top_k,
                    tfidf_weight=tfidf_weight,
                    wordnet_weight=wordnet_weight,
                    keyword_weight=keyword_weight
                )
                
                # Evaluate results
                eval_results = evaluate_classification(classified_df)
                
                # Calculate score
                score = calculate_objective_function(
                    eval_results, 
                    weight_coverage=objective_weights.get('coverage', 0.5),
                    weight_avg_labels=objective_weights.get('avg_labels', 0.2),
                    weight_avg_score=objective_weights.get('avg_score', 0.2),
                    weight_diversity=objective_weights.get('diversity', 0.1)
                )
                
                # Store parameters and results
                parameters = {
                    'threshold': threshold,
                    'top_k': top_k,
                    'tfidf_weight': tfidf_weight,
                    'wordnet_weight': wordnet_weight,
                    'keyword_weight': keyword_weight
                }
                
                run_results = {
                    'parameters': parameters,
                    'eval_results': eval_results,
                    'score': score
                }
                
                all_results.append(run_results)
                
                # Save results
                save_results(
                    classified_df, 
                    eval_results, 
                    parameters, 
                    score,
                    output_dir
                )
                
                # Update best configuration if better
                if score > best_score:
                    best_score = score
                    best_config = parameters
                    best_results = eval_results
                    best_classified_df = classified_df
                    logger.info(f"New best configuration found! Score: {best_score:.4f}")
    
    # Save best configuration
    if best_config:
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Best score: {best_score:.4f}")
        
        # Save the best results
        best_dir = save_results(
            best_classified_df,
            best_results,
            best_config,
            best_score,
            os.path.join(output_dir, 'best')
        )
        
        logger.info(f"Best results saved to {best_dir}")
        
        # Create a summary of all tested configurations
        create_comparison_report(all_results, os.path.join(output_dir, 'summary.csv'))
        
        # Create visualization of parameter impact
        create_parameter_impact_visualization(all_results, output_dir)
    
    return {
        'best_config': best_config,
        'best_score': best_score,
        'best_results': best_results,
        'all_results': all_results
    }

def create_comparison_report(all_results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Create a CSV report comparing all tested configurations.
    
    Args:
        all_results: List of results from all runs
        output_path: Path to save the CSV report
    """
    
    # Extract key metrics for each run
    rows = []
    
    for result in all_results:
        params = result['parameters']
        eval_metrics = result['eval_results']
        score = result['score']
        
        row = {
            'threshold': params.get('threshold'),
            'top_k': params.get('top_k'),
            'tfidf_weight': params.get('tfidf_weight'),
            'wordnet_weight': params.get('wordnet_weight'),
            'keyword_weight': params.get('keyword_weight'),
            'coverage_percentage': eval_metrics.get('coverage_percentage'),
            'avg_labels_per_company': eval_metrics.get('avg_labels_per_company'),
            'avg_labels_per_labeled_company': eval_metrics.get('avg_labels_per_labeled_company'),
            'avg_score': eval_metrics.get('avg_score'),
            'unique_labels_used': eval_metrics.get('unique_labels_used'),
            'diversity_ratio': eval_metrics.get('diversity_ratio'),
            'score': score
        }
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(by='score', ascending=False)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Comparison report saved to {output_path}")

def create_parameter_impact_visualization(all_results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Create visualizations showing the impact of different parameters on the score.
    
    Args:
        all_results: List of results from all runs
        output_dir: Directory to save visualizations
    """
    
    # Extract data
    thresholds = []
    top_ks = []
    tfidf_weights = []
    wordnet_weights = []
    keyword_weights = []
    scores = []
    coverages = []
    avg_labels = []
    
    for result in all_results:
        params = result['parameters']
        eval_metrics = result['eval_results']
        
        thresholds.append(params.get('threshold'))
        top_ks.append(params.get('top_k'))
        tfidf_weights.append(params.get('tfidf_weight'))
        wordnet_weights.append(params.get('wordnet_weight'))
        keyword_weights.append(params.get('keyword_weight'))
        scores.append(result['score'])
        coverages.append(eval_metrics.get('coverage_percentage'))
        avg_labels.append(eval_metrics.get('avg_labels_per_company'))
    
    # Create scatter plots for each parameter vs score
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, scores, alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Threshold on Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_impact.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(top_ks, scores, alpha=0.7)
    plt.xlabel('Top K')
    plt.ylabel('Score')
    plt.title('Impact of Top K on Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topk_impact.png'))
    plt.close()
    
    # Create 3D scatter plot for weights
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        p = ax.scatter(tfidf_weights, wordnet_weights, keyword_weights, c=scores, 
                    cmap='viridis', alpha=0.7, s=50)
        
        ax.set_xlabel('TF-IDF Weight')
        ax.set_ylabel('WordNet Weight')
        ax.set_zlabel('Keyword Weight')
        plt.colorbar(p, label='Score')
        plt.title('Impact of Weight Distribution on Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weights_impact_3d.png'))
        plt.close()
    except:
        logger.warning("Could not create 3D plot for weights")
    
    # Create a plot showing threshold vs coverage
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, coverages, alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Coverage Percentage')
    plt.title('Impact of Threshold on Coverage')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_coverage.png'))
    plt.close()
    
    # Create a plot showing threshold vs avg labels per company
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, avg_labels, alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Avg Labels per Company')
    plt.title('Impact of Threshold on Label Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_labels.png'))
    plt.close()

def main():
    """Main function to run threshold optimization"""
    parser = argparse.ArgumentParser(description='Optimize classification thresholds for insurance taxonomy')
    
    parser.add_argument('--preprocessed-companies', type=str, default='data/processed/processed_companies.csv',
                        help='Path to preprocessed companies CSV (default: data/processed/processed_companies.csv)')
    
    parser.add_argument('--preprocessed-taxonomy', type=str, default='data/processed/processed_taxonomy.csv',
                        help='Path to preprocessed taxonomy CSV (default: data/processed/processed_taxonomy.csv)')
    
    parser.add_argument('--output-dir', type=str, default='results/threshold_tests',
                        help='Directory to save test results (default: results/threshold_tests)')
    
    parser.add_argument('--min-threshold', type=float, default=0.05,
                        help='Minimum threshold to test (default: 0.05)')
    
    parser.add_argument('--max-threshold', type=float, default=0.2,
                        help='Maximum threshold to test (default: 0.2)')
    
    parser.add_argument('--threshold-steps', type=int, default=4,
                        help='Number of threshold steps to test (default: 4)')
    
    parser.add_argument('--top-k-values', type=str, default='3,5,10',
                        help='Comma-separated list of top-k values to test (default: 3,5,10)')
    
    parser.add_argument('--objective-coverage-weight', type=float, default=0.5,
                        help='Weight for coverage in objective function (default: 0.5)')
    
    parser.add_argument('--objective-avg-labels-weight', type=float, default=0.2,
                        help='Weight for average labels in objective function (default: 0.2)')
    
    parser.add_argument('--objective-avg-score-weight', type=float, default=0.2,
                        help='Weight for average score in objective function (default: 0.2)')
    
    parser.add_argument('--objective-diversity-weight', type=float, default=0.1,
                        help='Weight for diversity in objective function (default: 0.1)')
    
    args = parser.parse_args()
    
    logger.info("Starting threshold optimization for insurance taxonomy classification")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Make taxonomy_df accessible to evaluation function
    global taxonomy_df
    
    # Load preprocessed data
    companies_df, taxonomy_df = load_data(args.preprocessed_companies, args.preprocessed_taxonomy)
    
    # Generate threshold values
    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.threshold_steps)
    
    # Parse top-k values
    top_k_values = [int(k) for k in args.top_k_values.split(',')]
    
    # Define weight configurations to test
    weight_configs = [
        {'tfidf': 0.5, 'wordnet': 0.3, 'keyword': 0.2},
        {'tfidf': 0.6, 'wordnet': 0.2, 'keyword': 0.2},
        {'tfidf': 0.4, 'wordnet': 0.4, 'keyword': 0.2},
        {'tfidf': 0.7, 'wordnet': 0.2, 'keyword': 0.1},
        {'tfidf': 0.33, 'wordnet': 0.33, 'keyword': 0.34},
    ]
    
    # Define objective function weights
    objective_weights = {
        'coverage': args.objective_coverage_weight,
        'avg_labels': args.objective_avg_labels_weight,
        'avg_score': args.objective_avg_score_weight,
        'diversity': args.objective_diversity_weight
    }
    
    # Run the tests
    results = test_multiple_thresholds(
        companies_df,
        taxonomy_df,
        thresholds,
        top_k_values,
        weight_configs,
        objective_weights,
        args.output_dir
    )
    
    # Print the best configuration
    if results['best_config']:
        print("\n=== BEST CONFIGURATION ===")
        for param, value in results['best_config'].items():
            print(f"{param}: {value}")
        print(f"Score: {results['best_score']:.4f}")
        
        # Print key metrics
        print("\n=== KEY METRICS ===")
        best_results = results['best_results']
        print(f"Coverage: {best_results.get('coverage_percentage', 0):.2f}%")
        print(f"Avg labels per company: {best_results.get('avg_labels_per_company', 0):.2f}")
        print(f"Avg score: {best_results.get('avg_score', 0):.4f}")
        print(f"Unique labels used: {best_results.get('unique_labels_used', 0)}")
        
        print(f"\nComplete results available in: {os.path.join(args.output_dir, 'best')}")
        print(f"Summary of all tests: {os.path.join(args.output_dir, 'summary.csv')}")

if __name__ == "__main__":
    main()