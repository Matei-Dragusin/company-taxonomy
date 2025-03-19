#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Company data preprocessing module.
This module includes functionality for cleaning, transforming,
and preparing data for company taxonomy analysis.
"""

import os
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Union, Tuple, Optional
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing company data.
    """
    
    def __init__(self, raw_data_path: str = 'data/raw/', 
                 processed_data_path: str = 'data/processed/'):
        """
        Initialize the data processor.
        
        Args:
            raw_data_path: Path to raw data
            processed_data_path: Path where processed data will be saved
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # Create directory for processed data if it doesn't exist
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
            logger.info(f"Created directory {processed_data_path}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from raw data directory.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = os.path.join(self.raw_data_path, filename)
        logger.info(f"Loading data from {file_path}")
        
        # Determine file type by extension
        file_extension = os.path.splitext(filename)[1].lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
                
            logger.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_company_names(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Clean and standardize company names.
        
        Args:
            df: DataFrame containing company data
            column: Name of the column containing company names
            
        Returns:
            DataFrame with cleaned company names
        """
        logger.info(f"Cleaning company names in column: {column}")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert to string and strip whitespace
        df_clean[column] = df_clean[column].astype(str).str.strip()
        
        # Remove common legal suffixes
        legal_suffixes = r'\b(Inc|LLC|Ltd|Corp|Corporation|Limited|GmbH|PLC|LP|LLP|SA|AG|BV|SRL|SpA|NV|Co)\.?
        df_clean[column] = df_clean[column].str.replace(legal_suffixes, '', regex=True).str.strip()
        
        # Standardize case (Title Case)
        df_clean[column] = df_clean[column].str.title()
        
        # Remove special characters and extra spaces
        df_clean[column] = df_clean[column].str.replace(r'[^\w\s]', ' ', regex=True)
        df_clean[column] = df_clean[column].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        logger.info(f"Completed cleaning company names")
        
        return df_clean
    
    def extract_industry_keywords(self, df: pd.DataFrame, 
                                  description_column: str,
                                  industry_keywords: List[str] = None) -> pd.DataFrame:
        """
        Extract industry keywords from company descriptions.
        
        Args:
            df: DataFrame containing company data
            description_column: Name of the column containing company descriptions
            industry_keywords: List of industry keywords to look for (optional)
            
        Returns:
            DataFrame with added industry keyword columns
        """
        logger.info(f"Extracting industry keywords from column: {description_column}")
        
        df_result = df.copy()
        
        # Default industry keywords if not provided
        if industry_keywords is None:
            industry_keywords = [
                'technology', 'healthcare', 'finance', 'retail', 
                'manufacturing', 'energy', 'education', 'transportation',
                'media', 'telecommunications', 'hospitality', 'construction'
            ]
        
        # Create a column for each industry keyword
        for keyword in industry_keywords:
            column_name = f"is_{keyword}"
            df_result[column_name] = df_result[description_column].str.lower().str.contains(
                r'\b' + keyword + r'\b', 
                regex=True
            ).astype(int)
        
        # Create a combined industry column
        df_result['detected_industries'] = df_result.apply(
            lambda row: [keyword for keyword in industry_keywords 
                        if row[f"is_{keyword}"] == 1], 
            axis=1
        )
        
        logger.info(f"Added industry keyword detection columns")
        
        return df_result
    
    def categorize_by_size(self, df: pd.DataFrame, 
                          employees_column: str = 'employees',
                          revenue_column: str = None) -> pd.DataFrame:
        """
        Categorize companies by size based on number of employees and/or revenue.
        
        Args:
            df: DataFrame containing company data
            employees_column: Name of the column containing employee count
            revenue_column: Name of the column containing revenue (optional)
            
        Returns:
            DataFrame with added size category column
        """
        logger.info("Categorizing companies by size")
        
        df_result = df.copy()
        
        # Define size categories based on employee count
        conditions = [
            df_result[employees_column] < 10,
            df_result[employees_column].between(10, 49),
            df_result[employees_column].between(50, 249),
            df_result[employees_column].between(250, 999),
            df_result[employees_column] >= 1000
        ]
        
        categories = ['Micro', 'Small', 'Medium', 'Large', 'Enterprise']
        
        df_result['size_category'] = np.select(conditions, categories, default='Unknown')
        
        # If revenue information is available, refine the categorization
        if revenue_column and revenue_column in df_result.columns:
            logger.info("Refining size categories with revenue information")
            # Implement revenue-based categorization logic here
            pass
        
        logger.info("Completed company size categorization")
        
        return df_result
    
    def normalize_location_data(self, df: pd.DataFrame, 
                               location_column: str,
                               create_separate_columns: bool = True) -> pd.DataFrame:
        """
        Normalize location data to extract city, state/region, and country.
        
        Args:
            df: DataFrame containing company data
            location_column: Name of the column containing location information
            create_separate_columns: Whether to create separate columns for components
            
        Returns:
            DataFrame with normalized location data
        """
        logger.info(f"Normalizing location data from column: {location_column}")
        
        df_result = df.copy()
        
        # Simple pattern for locations like "City, State, Country" or "City, Country"
        pattern = r'^(?P<city>[^,]+)(?:,\s*(?P<state>[^,]+))?(?:,\s*(?P<country>.+))?
        
        if create_separate_columns:
            # Extract components using regex
            location_components = df_result[location_column].str.extract(pattern)
            
            # Add the extracted components to the dataframe
            df_result['city'] = location_components['city']
            df_result['state'] = location_components['state']
            df_result['country'] = location_components['country']
            
            # Fill NaN values in country if state is filled
            mask = df_result['country'].isna() & df_result['state'].notna()
            df_result.loc[mask, 'country'] = df_result.loc[mask, 'state']
            df_result.loc[mask, 'state'] = None
        
        logger.info("Completed location data normalization")
        
        return df_result
    
    def process_data(self, input_file: str, 
                    company_name_column: str,
                    description_column: str = None,
                    employees_column: str = None,
                    location_column: str = None,
                    output_file: str = None) -> pd.DataFrame:
        """
        Execute complete data preprocessing pipeline.
        
        Args:
            input_file: Name of the input file
            company_name_column: Column containing company names
            description_column: Column containing company descriptions (optional)
            employees_column: Column containing employee count (optional)
            location_column: Column containing location data (optional)
            output_file: Name of the output file (optional)
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting complete data preprocessing for {input_file}")
        
        # Load data
        df = self.load_data(input_file)
        
        # Clean company names
        df = self.clean_company_names(df, company_name_column)
        
        # Process description if column is provided
        if description_column and description_column in df.columns:
            df = self.extract_industry_keywords(df, description_column)
        
        # Categorize by size if employee column is provided
        if employees_column and employees_column in df.columns:
            df = self.categorize_by_size(df, employees_column)
        
        # Normalize location data if location column is provided
        if location_column and location_column in df.columns:
            df = self.normalize_location_data(df, location_column)
        
        # Save processed data if output file is specified
        if output_file:
            output_path = os.path.join(self.processed_data_path, output_file)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        
        logger.info("Data preprocessing completed successfully")
        
        return df


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Process sample data
    # preprocessor.process_data(
    #     input_file="companies.csv",
    #     company_name_column="name",
    #     description_column="description",
    #     employees_column="employee_count",
    #     location_column="headquarters",
    #     output_file="processed_companies.csv"
    # )