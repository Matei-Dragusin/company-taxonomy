"""
Company data preprocessing module.
This module includes functionality for cleaning, transforming,
and preparing data for company taxonomy analysis, with specific
features for insurance industry classification.
"""

import os
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Union, Tuple, Optional
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add industry-specific stopwords for insurance domain
INSURANCE_STOPWORDS = {
    'insurance', 'insure', 'policy', 'policies', 'coverage', 'claim', 'claims',
    'premium', 'premiums', 'risk', 'risks', 'underwrite', 'underwriting'
}

class DataPreprocessor:
    """
    Enhanced class for preprocessing company data with specific focus on
    insurance industry classification.
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
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union(INSURANCE_STOPWORDS)
    
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
    
    def load_company_and_taxonomy_data(self, 
                                     company_file: str, 
                                     taxonomy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both company data and insurance taxonomy data.
        
        Args:
            company_file: Name of the company data file
            taxonomy_file: Name of the taxonomy file
            
        Returns:
            Tuple containing company DataFrame and taxonomy DataFrame
        """
        companies_df = self.load_data(company_file)
        taxonomy_df = self.load_data(taxonomy_file)
        
        logger.info(f"Loaded {len(companies_df)} companies and {len(taxonomy_df)} taxonomy labels")
        
        return companies_df, taxonomy_df
    
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
        legal_suffixes = r'\b(Inc|LLC|Ltd|Corp|Corporation|Limited|GmbH|PLC|LP|LLP|SA|AG|BV|SRL|SpA|NV|Co)\.?'
        df_clean[column] = df_clean[column].str.replace(legal_suffixes, '', regex=True).str.strip()
        
        # Standardize case (Title Case)
        df_clean[column] = df_clean[column].str.title()
        
        # Remove special characters and extra spaces
        df_clean[column] = df_clean[column].str.replace(r'[^\w\s]', ' ', regex=True)
        df_clean[column] = df_clean[column].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        logger.info(f"Completed cleaning company names")
        
        return df_clean
    
    def normalize_industry_terms(self, text: str) -> str:
        """
        Normalize insurance industry-specific terms and abbreviations.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
            
        # Common insurance abbreviations and their full forms
        industry_terms = {
            'p&c': 'property and casualty',
            'wc': 'workers compensation',
            'gl': 'general liability',
            'pi': 'professional indemnity',
            'cyber': 'cybersecurity insurance',
            'ul': 'universal life',
            'ltc': 'long term care',
            'd&o': 'directors and officers',
            'a&h': 'accident and health',
            'e&o': 'errors and omissions',
            'b2b': 'business to business',
            'b2c': 'business to consumer'
        }
        
        for abbr, full_form in industry_terms.items():
            text = re.sub(rf'\b{abbr}\b', full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced version of text cleaning function.
        """
        
        if not isinstance(text, str):
            return ""
        
        # Normalize industry terms
        text = self.normalize_industry_terms(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Keep industry-specific stopwords
        insurance_terms = ['cyber-security', 'e-commerce', 'property & casualty', 'health & safety',
                       'business-to-business', 'business-to-consumer', 'third-party', 'third party']
    
        # Temporary replace insurance terms with placeholders
        placeholders = {}
        for i, term in enumerate(insurance_terms):
            if term in text.lower():
                placeholder = f"TERM_{i}_PLACEHOLDER"
                text = text.lower().replace(term, placeholder)
                placeholders[placeholder] = term.replace('-', ' ').replace('&', 'and')
                
        # Remove URL and mail references
        text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
        
            # Remove special characters but keep hyphens for compound words
        text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)
        
        # Replace hyphens with spaces to separate compound words
        text = text.replace('-', ' ')
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Reintroduce specific terms
        for placeholder, term in placeholders.items():
            text = text.replace(placeholder, term)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords but keep words relevant to insurance
        important_words = {'insurance', 'risk', 'liability', 'coverage', 'policy', 'claim', 'premium'}
        tokens = [token for token in tokens if token not in self.stop_words or token in important_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        if not isinstance(text, str) or text.strip() == "":
            return []
            
        tokens = text.split()
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
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
                'technology', 'healthcare', 'finance', 'insurance', 'retail', 
                'manufacturing', 'energy', 'education', 'transportation',
                'media', 'telecommunications', 'hospitality', 'construction',
                'liability', 'property', 'casualty', 'life', 'health',
                'commercial', 'residential', 'automotive', 'cyber', 'underwriting'
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
        
        # Check if employees column exists
        if employees_column not in df_result.columns:
            logger.warning(f"Employee column '{employees_column}' not found. Skipping size categorization.")
            return df_result
        
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
        
        # Check if location column exists
        if location_column not in df.columns:
            logger.warning(f"Location column '{location_column}' not found. Skipping location normalization.")
            return df
            
        df_result = df.copy()
        
        # Simple pattern for locations like "City, State, Country" or "City, Country"
        pattern = r'^(?P<city>[^,]+)(?:,\s*(?P<state>[^,]+))?(?:,\s*(?P<country>.+))?'
        
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
    
    def preprocess_company_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced preprocessing for company data, optimized for insurance taxonomy.
        
        Args:
            df: DataFrame containing company information
            
        Returns:
            Preprocessed DataFrame with enhanced features
        """
        logger.info("Starting enhanced company data preprocessing")
        
        df_result = df.copy()
        
        # Clean company descriptions
        if 'description' in df_result.columns:
            logger.info("Processing company descriptions")
            df_result['cleaned_description'] = df_result['description'].apply(self.clean_text)
            
            # Add bigrams and trigrams as features
            df_result['description_bigrams'] = df_result['cleaned_description'].apply(
                lambda x: self.extract_ngrams(x, 2)
            )
            df_result['description_trigrams'] = df_result['cleaned_description'].apply(
                lambda x: self.extract_ngrams(x, 3)
            )
        
        # Process business tags
        if 'business_tags' in df_result.columns:
            logger.info("Processing business tags")
            df_result['cleaned_tags'] = df_result['business_tags'].apply(self.clean_text)
        
        # Process sector information
        if 'sector' in df_result.columns:
            logger.info("Processing sector information")
            df_result['cleaned_sector'] = df_result['sector'].apply(self.clean_text)
        
        # Process category information
        if 'category' in df_result.columns:
            logger.info("Processing category information")
            df_result['cleaned_category'] = df_result['category'].apply(self.clean_text)
            
        # Process niche information
        if 'niche' in df_result.columns:
            logger.info("Processing niche information")
            df_result['cleaned_niche'] = df_result['niche'].apply(self.clean_text)
        
        # Create weighted features for better classification
        text_columns = [col for col in ['cleaned_description', 'cleaned_tags', 'cleaned_sector', 
                                        'cleaned_category', 'cleaned_niche'] if col in df_result.columns]
        
        # Define weights for different text columns (higher weight = more important)
        weights = {
            'cleaned_description': 1.0, 
            'cleaned_tags': 1.5, 
            'cleaned_sector': 2.0,
            'cleaned_category': 2.0,
            'cleaned_niche': 2.5
        }
        
        # Ensure all needed columns are properly formatted
        for col in text_columns:
            if not all(isinstance(x, str) for x in df_result[col]):
                df_result[col] = df_result[col].fillna("").astype(str)
        
        # Create combined weighted features text
        logger.info("Creating combined weighted features")
        df_result['combined_features'] = df_result[text_columns].apply(
            lambda row: ' '.join([
                ' '.join([text] * int(weights.get(col, 1.0) * 10)) 
                for col, text in row.items() if isinstance(text, str) and text.strip() != ""
            ]),
            axis=1
        )
        
        # Extract industry keywords
        if 'cleaned_description' in df_result.columns:
            df_result = self.extract_industry_keywords(df_result, 'cleaned_description')
        
        logger.info("Enhanced company preprocessing completed")
        
        return df_result
    
    def preprocess_taxonomy(self, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess insurance taxonomy for classification.
        
        Args:
            taxonomy_df: DataFrame containing insurance taxonomy
            
        Returns:
            Preprocessed taxonomy DataFrame
        """
        logger.info("Starting taxonomy preprocessing")
        
        df_result = taxonomy_df.copy()
        
        # Clean taxonomy labels
        if 'label' in df_result.columns:
            logger.info("Processing taxonomy labels")
            df_result['cleaned_label'] = df_result['label'].apply(self.clean_text)
            
            # Extract keywords from labels
            df_result['label_keywords'] = df_result['cleaned_label'].apply(
                lambda x: [word for word in x.split() if len(word) > 3]
            )
            
            # Extract bigrams from labels
            df_result['label_bigrams'] = df_result['cleaned_label'].apply(
                lambda x: self.extract_ngrams(x, 2)
            )
        
        # Process descriptions if available
        if 'description' in df_result.columns:
            logger.info("Processing taxonomy descriptions")
            df_result['cleaned_description'] = df_result['description'].apply(self.clean_text)
        
        logger.info("Taxonomy preprocessing completed")
        
        return df_result
    
    def process_data(self, input_file: str, 
                    company_name_column: str = None,
                    description_column: str = 'description',
                    employees_column: str = None,
                    location_column: str = None,
                    output_file: str = None,
                    enhanced_processing: bool = True) -> pd.DataFrame:
        """
        Execute complete data preprocessing pipeline.
        
        Args:
            input_file: Name of the input file
            company_name_column: Column containing company names (optional)
            description_column: Column containing company descriptions
            employees_column: Column containing employee count (optional)
            location_column: Column containing location data (optional)
            output_file: Name of the output file (optional)
            enhanced_processing: Whether to apply enhanced processing for insurance taxonomy
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting complete data preprocessing for {input_file}")
        
        # Load data
        df = self.load_data(input_file)
        
        # Clean company names if column is provided
        if company_name_column and company_name_column in df.columns:
            df = self.clean_company_names(df, company_name_column)
        
        # Apply enhanced preprocessing for insurance taxonomy if requested
        if enhanced_processing:
            df = self.preprocess_company_data(df)
        else:
            # Process description if column is provided (basic processing)
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
    
    def process_company_and_taxonomy(self, 
                                  company_file: str, 
                                  taxonomy_file: str,
                                  output_company_file: str = "processed_companies.csv",
                                  output_taxonomy_file: str = "processed_taxonomy.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process both company and taxonomy data for insurance classification.
        
        Args:
            company_file: Name of the company data file
            taxonomy_file: Name of the taxonomy file
            output_company_file: Name of the output company file
            output_taxonomy_file: Name of the output taxonomy file
            
        Returns:
            Tuple containing processed company DataFrame and taxonomy DataFrame
        """
        logger.info(f"Starting combined processing for companies and taxonomy")
        
        # Load data
        companies_df, taxonomy_df = self.load_company_and_taxonomy_data(company_file, taxonomy_file)
        
        # Process company data with enhanced processing
        processed_companies = self.preprocess_company_data(companies_df)
        
        # Process taxonomy data
        processed_taxonomy = self.preprocess_taxonomy(taxonomy_df)
        
        # Save processed company data
        if output_company_file:
            output_path = os.path.join(self.processed_data_path, output_company_file)
            processed_companies.to_csv(output_path, index=False)
            logger.info(f"Saved processed company data to {output_path}")
            
        # Save processed taxonomy data
        if output_taxonomy_file:
            output_path = os.path.join(self.processed_data_path, output_taxonomy_file)
            processed_taxonomy.to_csv(output_path, index=False)
            logger.info(f"Saved processed taxonomy data to {output_path}")
        
        logger.info("Combined processing completed successfully")
        
        return processed_companies, processed_taxonomy
    
    # Adaugă această metodă în clasa DataPreprocessor din src/preprocessing/preprocessing.py:

    def expand_taxonomy_labels(self, taxonomy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extinde etichetele taxonomiei pentru a îmbunătăți potrivirea.
        
        Args:
            taxonomy_df: DataFrame cu taxonomia de asigurări
        
        Returns:
            DataFrame cu etichete expandate
        """
        logger.info("Expandarea etichetelor taxonomiei pentru îmbunătățirea potrivirii")
        
        df_result = taxonomy_df.copy()
        
        # Adaugă sinonime și variații pentru etichetele existente
        if 'label' in df_result.columns:
            # Creează un dicționar de sinonime pentru domenii comune
            industry_synonyms = {
                'Real Estate': ['property', 'realty', 'housing', 'buildings', 'land'],
                'Construction': ['building', 'contracting', 'development', 'infrastructure'],
                'Financial': ['finance', 'banking', 'investment', 'monetary', 'fiscal'],
                'Insurance': ['insurer', 'coverage', 'underwriting', 'policy', 'risk management'],
                'Manufacturing': ['production', 'fabrication', 'industrial', 'factory'],
                'Technology': ['tech', 'IT', 'digital', 'software', 'computing', 'information systems'],
                'Healthcare': ['medical', 'health', 'clinical', 'patient care', 'wellness'],
                'Transportation': ['logistics', 'shipping', 'freight', 'transit', 'fleet'],
                'Consulting': ['advisory', 'counseling', 'professional services'],
                'Retail': ['sales', 'commerce', 'merchandising', 'store', 'shop']
            }
            
            # Adaugă coloană pentru termeni expandați
            df_result['expanded_terms'] = df_result['label'].apply(
                lambda label: self._expand_label_with_synonyms(label, industry_synonyms)
            )
            
            # Adaugă coloană pentru descriere expandată
            if 'description' in df_result.columns:
                df_result['expanded_description'] = df_result.apply(
                    lambda row: f"{row['label']} {row['description'] if pd.notnull(row['description']) else ''} {row['expanded_terms']}",
                    axis=1
                )
            else:
                df_result['expanded_description'] = df_result.apply(
                    lambda row: f"{row['label']} {row['expanded_terms']}",
                    axis=1
                )
            
            # Curăță descrierea expandată
            df_result['cleaned_expanded'] = df_result['expanded_description'].apply(self.clean_text)
        
        logger.info("Expandare taxonomie completă")
        
        return df_result

    def _expand_label_with_synonyms(self, label: str, industry_synonyms: Dict[str, List[str]]) -> str:
        """
        Expandează o etichetă cu sinonime relevante.
        
        Args:
            label: Eticheta originală
            industry_synonyms: Dicționar de sinonime pentru industrii
            
        Returns:
            String cu termeni expandați
        """
        expanded_terms = []
        
        # Adaugă variații pentru eticheta originală
        label_parts = label.split()
        
        # Verifică pentru fiecare cuvânt cheie din dicționarul de sinonime
        for key, synonyms in industry_synonyms.items():
            if key.lower() in label.lower():
                expanded_terms.extend(synonyms)
        
        # Adaugă variații de fraze din etichetă
        for i in range(len(label_parts)):
            for j in range(i+1, len(label_parts)+1):
                phrase = ' '.join(label_parts[i:j])
                if len(phrase.split()) > 1:  # Doar fraze, nu cuvinte individuale
                    expanded_terms.append(phrase)
        
        return ' '.join(expanded_terms)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Process company and taxonomy data
    preprocessor.process_company_and_taxonomy(
        company_file="companies.csv",
        taxonomy_file="insurance_taxonomy.csv",
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )