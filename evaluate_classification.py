#!/usr/bin/env python
"""
Script for evaluating company classification performance
and analyzing the results.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from typing import Dict, List, Tuple
from collections import Counter

# Add the project's root directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_classification_results(file_path: str) -> pd.DataFrame:
    """
    Load classification results from a CSV file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        DataFrame with classification results
    """
    logger.info(f"Loading classification results from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist!")
        return None
    
    try:
        results_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(results_df)} records from the results file")
        return results_df
    except Exception as e:
        logger.error(f"Error loading the results file: {e}")
        return None

def analyze_label_distribution(results_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of assigned labels.
    
    Args:
        results_df: DataFrame with classification results
        
    Returns:
        Dictionary with statistics about label distribution
    """
    logger.info("Analyzing label distribution")
    
    # Convert columns of lists represented as strings into Python lists
    if 'insurance_labels' not in results_df.columns and 'insurance_label' in results_df.columns:
        # If we only have insurance_label, split it into a list
        results_df['insurance_labels'] = results_df['insurance_label'].str.split(', ')
    
    # Check if we have access to the label lists
    if 'insurance_labels' not in results_df.columns:
        logger.error("The column 'insurance_labels' does not exist in the results!")
        return {}
    
    # Count all labels
    all_labels = []
    for labels in results_df['insurance_labels']:
        if isinstance(labels, list):
            all_labels.extend(labels)
        elif isinstance(labels, str):
            if labels.startswith('[') and labels.endswith(']'):
                # Convert string representation of a list into a Python list
                try:
                    parsed_labels = eval(labels)
                    if isinstance(parsed_labels, list):
                        all_labels.extend(parsed_labels)
                except:
                    pass
            else:
                # Possibly a comma-separated list
                label_items = [item.strip() for item in labels.split(',') if item.strip()]
                all_labels.extend(label_items)
    
    label_counts = Counter(all_labels)
    
    # Companies per label
    label_company_counts = {}
    for label in set(all_labels):
        companies_with_label = 0
        for labels in results_df['insurance_labels']:
            if isinstance(labels, list) and label in labels:
                companies_with_label += 1
            elif isinstance(labels, str):
                if label in labels:  # Simple check
                    companies_with_label += 1
        label_company_counts[label] = companies_with_label
    
    # Label distribution statistics
    total_companies = len(results_df)
    total_label_assignments = len(all_labels)
    unique_labels = len(label_counts)
    avg_labels_per_company = total_label_assignments / total_companies if total_companies > 0 else 0
    
    # Top and least frequent labels
    top_labels = label_counts.most_common(10)
    bottom_labels = label_counts.most_common()[:-11:-1]
    
    return {
        'total_companies': total_companies,
        'total_label_assignments': total_label_assignments,
        'unique_labels': unique_labels,
        'avg_labels_per_company': avg_labels_per_company,
        'label_counts': dict(label_counts),
        'label_company_counts': label_company_counts,
        'top_labels': top_labels,
        'bottom_labels': bottom_labels
    }

def analyze_similarity_scores(results_df: pd.DataFrame) -> Dict:
    """
    Analyze similarity scores for assigned labels.
    
    Args:
        results_df: DataFrame with classification results
        
    Returns:
        Dictionary with statistics about similarity scores
    """
    logger.info("Analyzing similarity scores")
    
    # Check if we have access to similarity scores
    if 'insurance_label_scores' not in results_df.columns:
        logger.error("The column 'insurance_label_scores' does not exist in the results!")
        return {}
    
    # Collect all similarity scores
    all_scores = []
    for scores in results_df['insurance_label_scores']:
        if isinstance(scores, list):
            all_scores.extend(scores)
        elif isinstance(scores, str) and scores.startswith('[') and scores.endswith(']'):
            try:
                parsed_scores = eval(scores)
                if isinstance(parsed_scores, list):
                    all_scores.extend(parsed_scores)
            except:
                pass
    
    if not all_scores:
        logger.warning("No similarity scores found for analysis")
        return {}
    
    # Calculate statistics about scores
    avg_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    # Score distribution
    score_ranges = {
        '0.0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
        '0.9-1.0': 0
    }
    
    for score in all_scores:
        for range_str in score_ranges:
            lower, upper = map(float, range_str.split('-'))
            if lower <= score < upper or (upper == 1.0 and score == 1.0):
                score_ranges[range_str] += 1
                break
    
    return {
        'avg_score': avg_score,
        'median_score': median_score,
        'min_score': min_score,
        'max_score': max_score,
        'score_ranges': score_ranges
    }

def analyze_company_coverage(results_df: pd.DataFrame) -> Dict:
    """
    Analyze company coverage - how many companies received labels.
    
    Args:
        results_df: DataFrame with classification results
        
    Returns:
        Dictionary with statistics about company coverage
    """
    logger.info("Analyzing company coverage")
    
    total_companies = len(results_df)
    
    # Check how many companies received at least one label
    companies_with_labels = 0
    companies_without_labels = 0
    
    for labels in results_df['insurance_labels']:
        if isinstance(labels, list) and labels:
            companies_with_labels += 1
        elif isinstance(labels, str):
            if labels and labels != '[]' and labels != 'Unclassified':
                companies_with_labels += 1
            else:
                companies_without_labels += 1
        else:
            companies_without_labels += 1
    
    # Distribution of the number of labels per company
    label_count_distribution = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}
    
    for labels in results_df['insurance_labels']:
        count = 0
        if isinstance(labels, list):
            count = len(labels)
        elif isinstance(labels, str) and labels.startswith('[') and labels.endswith(']'):
            try:
                parsed_labels = eval(labels)
                if isinstance(parsed_labels, list):
                    count = len(parsed_labels)
            except:
                pass
        elif isinstance(labels, str) and labels != 'Unclassified':
            count = len([l.strip() for l in labels.split(',') if l.strip()])
        
        if count >= 4:
            label_count_distribution['4+'] += 1
        else:
            label_count_distribution[count] += 1
    
    coverage_percentage = (companies_with_labels / total_companies * 100) if total_companies > 0 else 0
    
    return {
        'total_companies': total_companies,
        'companies_with_labels': companies_with_labels,
        'companies_without_labels': companies_without_labels,
        'coverage_percentage': coverage_percentage,
        'label_count_distribution': label_count_distribution
    }

def plot_label_distribution(distribution_stats: Dict, output_dir: str = 'results/figures/'):
    """
    Generate plots for label distribution.
    
    Args:
        distribution_stats: Label distribution statistics
        output_dir: Directory where plots will be saved
    """
    logger.info("Generating plots for label distribution")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot the distribution of the most frequent labels
    plt.figure(figsize=(12, 6))
    
    if 'top_labels' in distribution_stats and distribution_stats['top_labels']:
        labels = [label for label, count in distribution_stats['top_labels']]
        counts = [count for label, count in distribution_stats['top_labels']]
        
        # Use range for x and add labels manually
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.title('Top 10 Most Frequent Labels')
        plt.xlabel('Label')
        plt.ylabel('Number of Assignments')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'top_labels.png'))
    
    # Plot the distribution of the number of labels per company
    plt.figure(figsize=(10, 6))
    
    label_counts = distribution_stats.get('label_count_distribution', {})
    if label_counts:
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Convert labels to string for safety
        str_labels = [str(label) for label in labels]
        
        # Use range for x and add labels manually
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, str_labels)
        plt.title('Distribution of Number of Labels per Company')
        plt.xlabel('Number of Labels')
        plt.ylabel('Number of Companies')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'label_count_distribution.png'))
    
    # Plot the distribution of similarity scores
    plt.figure(figsize=(10, 6))
    
    score_ranges = distribution_stats.get('score_ranges', {})
    if score_ranges:
        ranges = list(score_ranges.keys())
        counts = list(score_ranges.values())
        
        # Use range for x and add labels manually
        x_pos = np.arange(len(ranges))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, ranges, rotation=45)
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Score Range')
        plt.ylabel('Number of Assignments')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'similarity_score_distribution.png'))
    
    logger.info(f"Plots have been saved in the directory {output_dir}")

def generate_evaluation_report(results_df: pd.DataFrame, output_file: str = 'results/evaluation_report.txt'):
    """
    Generate a detailed evaluation report.
    
    Args:
        results_df: DataFrame with classification results
        output_file: Output file for the report
    """
    logger.info("Generating evaluation report")
    
    # Create the directory for the report if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect all statistics
    label_stats = analyze_label_distribution(results_df)
    score_stats = analyze_similarity_scores(results_df)
    coverage_stats = analyze_company_coverage(results_df)
    
    # Generate plots
    plot_label_distribution({**label_stats, **score_stats, **coverage_stats})
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write("=== CLASSIFICATION EVALUATION REPORT ===\n\n")
        
        f.write("== GENERAL STATISTICS ==\n")
        f.write(f"Total number of companies: {coverage_stats['total_companies']}\n")
        f.write(f"Companies with assigned labels: {coverage_stats['companies_with_labels']} ({coverage_stats['coverage_percentage']:.2f}%)\n")
        f.write(f"Companies without labels: {coverage_stats['companies_without_labels']}\n")
        f.write(f"Average number of labels per company: {label_stats['avg_labels_per_company']:.2f}\n\n")
        
        f.write("== DISTRIBUTION OF NUMBER OF LABELS PER COMPANY ==\n")
        for label_count, num_companies in coverage_stats['label_count_distribution'].items():
            f.write(f"{label_count} labels: {num_companies} companies\n")
        f.write("\n")
        
        f.write("== LABEL STATISTICS ==\n")
        f.write(f"Total number of label assignments: {label_stats['total_label_assignments']}\n")
        f.write(f"Number of unique labels used: {label_stats['unique_labels']}\n\n")
        
        f.write("== TOP 10 MOST FREQUENT LABELS ==\n")
        for label, count in label_stats['top_labels']:
            f.write(f"{label}: {count} assignments\n")
        f.write("\n")
        
        f.write("== SIMILARITY SCORES ==\n")
        f.write(f"Average similarity score: {score_stats['avg_score']:.4f}\n")
        f.write(f"Median score: {score_stats['median_score']:.4f}\n")
        f.write(f"Minimum score: {score_stats['min_score']:.4f}\n")
        f.write(f"Maximum score: {score_stats['max_score']:.4f}\n\n")
        
        f.write("== DISTRIBUTION OF SIMILARITY SCORES ==\n")
        for range_str, count in score_stats['score_ranges'].items():
            f.write(f"Range {range_str}: {count} assignments\n")
        
        f.write("\n=== END OF REPORT ===\n")
    
    logger.info(f"The evaluation report has been generated at {output_file}")
    return {**label_stats, **score_stats, **coverage_stats}

def main():
    """Main function for running the evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate company classification performance')
    
    parser.add_argument('--input-file', type=str, default='data/processed/classified_companies.csv',
                        help='Path to the classification results file (default: data/processed/classified_companies.csv)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    logger.info("Starting classification performance evaluation")
    
    # Load classification results
    results_df = load_classification_results(args.input_file)
    
    if results_df is None:
        logger.error("Failed to load classification results. Exiting program.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output directory: {args.output_dir}")
    
    # Generate evaluation report
    output_file = os.path.join(args.output_dir, 'evaluation_report.txt')
    evaluation_stats = generate_evaluation_report(results_df, output_file)
    
    # Display some key statistics
    print("\n=== KEY STATISTICS ===")
    print(f"Total number of companies: {evaluation_stats['total_companies']}")
    print(f"Coverage: {evaluation_stats['coverage_percentage']:.2f}% of companies received labels")
    print(f"Average number of labels per company: {evaluation_stats['avg_labels_per_company']:.2f}")
    print(f"Average similarity score: {evaluation_stats['avg_score']:.4f}")
    print(f"Most frequent label: {evaluation_stats['top_labels'][0][0]} ({evaluation_stats['top_labels'][0][1]} assignments)")
    print(f"\nThe full report and plots are available in: {args.output_dir}")

if __name__ == "__main__":
    main()