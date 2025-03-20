#!/usr/bin/env python
"""
Script to run the combined company data preprocessing pipeline.
This script uses the integrated DataPreprocessor class to process
both company data and insurance taxonomy data.
"""

import os
import sys
from typing import Tuple

# Add the project root directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import preprocessing module (updated path)
from src.preprocessing.preprocessing import DataPreprocessor

def ensure_directories_exist():
    """Ensure necessary data directories exist"""
    dirs = ['data/raw', 'data/processed']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def main():
    """Main function to run the preprocessing pipeline"""
    print("Starting company taxonomy preprocessing pipeline...")
    
    # Ensure data directories exist
    ensure_directories_exist()
    
    # Define file paths
    companies_file = 'companies.csv'
    taxonomy_file = 'insurance_taxonomy.csv'
    
    # Check if input files exist in raw data directory
    raw_data_path = 'data/raw/'
    if not os.path.exists(os.path.join(raw_data_path, companies_file)):
        print(f"Error: Companies file not found at {os.path.join(raw_data_path, companies_file)}")
        print("Please make sure to place companies.csv in the data/raw directory.")
        return
    
    if not os.path.exists(os.path.join(raw_data_path, taxonomy_file)):
        print(f"Error: Taxonomy file not found at {os.path.join(raw_data_path, taxonomy_file)}")
        print("Please make sure to place insurance_taxonomy.csv in the data/raw directory.")
        return
    
    print(f"\nInitializing DataPreprocessor...")
    
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    print(f"\nProcessing company and taxonomy data...")
    
    # Process company and taxonomy data together
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file=companies_file,
        taxonomy_file=taxonomy_file,
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )
    
    print(f"\nPreprocessing Pipeline Complete")
    print(f"Processed companies data shape: {companies_df.shape}")
    print(f"Processed taxonomy data shape: {taxonomy_df.shape}")
    print(f"All processed files are available in the data/processed directory:")
    print(f"  - data/processed/processed_companies.csv")
    print(f"  - data/processed/processed_taxonomy.csv")

if __name__ == "__main__":
    main()