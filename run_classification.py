#!/usr/bin/env python
"""
Script pentru rularea clasificării companiilor în taxonomia de asigurări
folosind tehnica TF-IDF.
"""

import os
import sys
import pandas as pd
import logging
import argparse
from typing import Tuple

# Adăugăm directorul rădăcină al proiectului în path pentru a permite importuri
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importăm modulele necesare
from src.preprocessing.preprocessing import DataPreprocessor
from src.feature_engineering.tfidf_processor import TFIDFProcessor

# Configurăm logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directories_exist():
    """Asigură existența directoarelor necesare"""
    dirs = ['data/raw', 'data/processed', 'models']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"S-a creat directorul: {dir_path}")

def load_and_preprocess_data(company_file: str, taxonomy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Încarcă și preprocesează datele companiilor și taxonomiei.
    
    Args:
        company_file: Numele fișierului cu companiile
        taxonomy_file: Numele fișierului cu taxonomia
        
    Returns:
        Tuple cu DataFrame-urile preprocesate pentru companii și taxonomie
    """
    logger.info(f"Încărcare și preprocesare date din {company_file} și {taxonomy_file}")
    
    # Inițializăm preprocesorul
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Procesăm datele
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file=company_file,
        taxonomy_file=taxonomy_file,
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )
    
    return companies_df, taxonomy_df

def run_classification(companies_df: pd.DataFrame, 
                      taxonomy_df: pd.DataFrame, 
                      top_k: int = 3, 
                      threshold: float = 0.1,
                      output_file: str = "classified_companies.csv") -> pd.DataFrame:
    """
    Rulează clasificarea companiilor folosind TF-IDF.
    
    Args:
        companies_df: DataFrame cu datele companiilor preprocesate
        taxonomy_df: DataFrame cu taxonomia preprocesată
        top_k: Numărul maxim de etichete de asignat unei companii
        threshold: Pragul minim de similitudine pentru a asigna o etichetă
        output_file: Numele fișierului pentru salvarea rezultatelor
        
    Returns:
        DataFrame cu companiile clasificate
    """
    logger.info("Începerea clasificării companiilor folosind TF-IDF")
    
    # Inițializăm procesorul TF-IDF
    tfidf_processor = TFIDFProcessor(
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
        models_path='models/'
    )
    
    # Antrenăm vectorizorul pe datele combinate
    tfidf_processor.fit_vectorizer(
        companies_df, 
        taxonomy_df, 
        company_text_column='combined_features', 
        taxonomy_column='cleaned_label'
    )
    
    # Clasificăm companiile
    classified_companies = tfidf_processor.assign_insurance_labels(
        companies_df,
        taxonomy_df,
        top_k=top_k,
        threshold=threshold
    )
    
    # Salvăm modelele pentru utilizare ulterioară
    tfidf_processor.save_models()
    
    # Salvăm rezultatele
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"Rezultatele clasificării au fost salvate în {output_path}")
    
    return classified_companies

def main():
    """Funcție principală pentru rularea clasificării"""
    parser = argparse.ArgumentParser(description='Clasificare companii pentru taxonomia de asigurări')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Numele fișierului cu lista companiilor (implicit: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Numele fișierului cu taxonomia (implicit: insurance_taxonomy.csv)')
    
    parser.add_argument('--top-k', type=int, default=3,
                        help='Numărul maxim de etichete de asignat unei companii (implicit: 3)')
    
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Pragul minim de similitudine pentru a asigna o etichetă (implicit: 0.1)')
    
    parser.add_argument('--output-file', type=str, default='classified_companies.csv',
                        help='Numele fișierului de ieșire (implicit: classified_companies.csv)')
    
    args = parser.parse_args()
    
    logger.info("Pornirea clasificării companiilor pentru taxonomia de asigurări")
    
    # Asigurăm existența directoarelor necesare
    ensure_directories_exist()
    
    # Verificăm existența fișierelor de intrare
    raw_data_path = 'data/raw/'
    if not os.path.exists(os.path.join(raw_data_path, args.company_file)):
        logger.error(f"Eroare: Fișierul companiilor nu a fost găsit la {os.path.join(raw_data_path, args.company_file)}")
        return
    
    if not os.path.exists(os.path.join(raw_data_path, args.taxonomy_file)):
        logger.error(f"Eroare: Fișierul taxonomiei nu a fost găsit la {os.path.join(raw_data_path, args.taxonomy_file)}")
        return
    
    # Încărcăm și preprocesăm datele
    companies_df, taxonomy_df = load_and_preprocess_data(args.company_file, args.taxonomy_file)
    
    # Rulăm clasificarea
    classified_companies = run_classification(
        companies_df,
        taxonomy_df,
        top_k=args.top_k,
        threshold=args.threshold,
        output_file=args.output_file
    )
    
    # Afișăm statistici despre rezultate
    total_companies = len(classified_companies)
    classified_count = sum(1 for labels in classified_companies['insurance_labels'] if labels)
    avg_labels_per_company = sum(len(labels) for labels in classified_companies['insurance_labels']) / total_companies
    
    logger.info(f"Clasificare completă:")
    logger.info(f"  - Total companii procesate: {total_companies}")
    logger.info(f"  - Companii clasificate cu succes: {classified_count} ({classified_count/total_companies*100:.1f}%)")
    logger.info(f"  - Număr mediu de etichete per companie: {avg_labels_per_company:.2f}")
    logger.info(f"Rezultatele complete sunt disponibile în: data/processed/{args.output_file}")

if __name__ == "__main__":
    main()