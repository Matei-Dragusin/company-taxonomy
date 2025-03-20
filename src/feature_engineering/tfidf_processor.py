"""
Module pentru implementarea TF-IDF și feature engineering
pentru clasificarea companiilor în taxonomia de asigurări.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import joblib
import os

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TFIDFProcessor:
    """
    Clasă pentru generarea și gestionarea reprezentărilor TF-IDF
    pentru textele companiilor și taxonomia de asigurări.
    """
    
    def __init__(self, min_df: int = 2, max_df: float = 0.85, 
                 ngram_range: Tuple[int, int] = (1, 2), models_path: str = 'models/'):
        """
        Inițializează procesorul TF-IDF.
        
        Args:
            min_df: Frecvența minimă a documentelor pentru a include un termen
            max_df: Frecvența maximă a documentelor pentru a include un termen (eliminarea cuvintelor prea comune)
            ngram_range: Intervalul de n-grame pentru vectorizare (1,2 = unigrame și bigrame)
            models_path: Calea unde vor fi salvate modelele antrenate
        """
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.models_path = models_path
        
        # Creare director pentru modele dacă nu există
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            logger.info(f"S-a creat directorul {models_path}")
        
        # Inițializăm un singur vectorizor TF-IDF comun pentru companii și taxonomie
        # Aceasta va asigura că toți vectorii sunt în același spațiu vectorial
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=True  # Aplicarea scalării logaritmice pentru atenuarea frecvenței termenilor
        )
    
    def fit_vectorizer(self, companies_df: pd.DataFrame, taxonomy_df: pd.DataFrame, 
                      company_text_column: str = 'combined_features',
                      taxonomy_column: str = 'cleaned_label') -> None:
        """
        Antrenează vectorizorul TF-IDF pe datele combinate ale companiilor și taxonomiei.
        Aceasta asigură că toți vectorii vor fi în același spațiu vectorial.
        
        Args:
            companies_df: DataFrame cu datele companiei preprocesate
            taxonomy_df: DataFrame cu taxonomia de asigurări preprocesată
            company_text_column: Coloana din companies_df conținând textul pentru vectorizare
            taxonomy_column: Coloana din taxonomy_df conținând textul pentru vectorizare
        """
        logger.info("Antrenare vectorizor TF-IDF pe date combinate (companii + taxonomie)")
        
        # Verificăm existența coloanelor necesare
        if company_text_column not in companies_df.columns:
            logger.error(f"Coloana {company_text_column} nu există în datasetul companiilor!")
            return
            
        if taxonomy_column not in taxonomy_df.columns:
            logger.error(f"Coloana {taxonomy_column} nu există în datasetul taxonomiei!")
            return
        
        # Combinăm textele din companiile și taxonomie pentru antrenarea vectorizorului
        company_texts = companies_df[company_text_column].fillna('').tolist()
        taxonomy_texts = taxonomy_df[taxonomy_column].fillna('').tolist()
        
        # Combinăm toate textele și antrenăm vectorizorul pe ele
        all_texts = company_texts + taxonomy_texts
        logger.info(f"Antrenare vectorizor pe {len(all_texts)} texte combinate")
        
        self.vectorizer.fit(all_texts)
        
        # Salvăm dimensiunea vocabularului
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vocabular creat cu {self.vocabulary_size} termeni")
    
    def transform_text(self, texts: List[str]) -> np.ndarray:
        """
        Transformă texte în reprezentări TF-IDF.
        
        Args:
            texts: Lista de texte pentru a fi transformate
            
        Returns:
            Matrice TF-IDF pentru textele date
        """
        if not hasattr(self, 'vectorizer') or not hasattr(self.vectorizer, 'vocabulary_'):
            logger.error("Vectorizorul nu a fost antrenat! Apelați mai întâi fit_vectorizer().")
            return None
            
        return self.vectorizer.transform(texts)
    
    def transform_company_data(self, df: pd.DataFrame, text_column: str = 'combined_features') -> np.ndarray:
        """
        Transformă datele companiei în reprezentări TF-IDF.
        
        Args:
            df: DataFrame cu datele companiei preprocesate
            text_column: Coloana conținând textul pentru transformare
            
        Returns:
            Matrice TF-IDF pentru companiile date
        """
        logger.info(f"Transformare date companie din coloana {text_column} în reprezentări TF-IDF")
        
        if text_column not in df.columns:
            logger.error(f"Coloana {text_column} nu există în datasetul companiilor!")
            return None
        
        company_texts = df[text_column].fillna('').tolist()
        company_vectors = self.transform_text(company_texts)
        
        logger.info(f"Transformare completă - forma matricei: {company_vectors.shape}")
        return company_vectors
    
    def transform_taxonomy(self, taxonomy_df: pd.DataFrame, column: str = 'cleaned_label') -> np.ndarray:
        """
        Transformă etichetele taxonomiei în reprezentări TF-IDF.
        
        Args:
            taxonomy_df: DataFrame cu taxonomia de asigurări preprocesată
            column: Coloana conținând etichetele curățate
            
        Returns:
            Matrice TF-IDF pentru etichetele taxonomiei
        """
        logger.info(f"Transformare etichete taxonomie din coloana {column}")
        
        if column not in taxonomy_df.columns:
            logger.error(f"Coloana {column} nu există în datasetul taxonomiei!")
            return None
        
        taxonomy_texts = taxonomy_df[column].fillna('').tolist()
        taxonomy_vectors = self.transform_text(taxonomy_texts)
        
        logger.info(f"Transformare completă - forma matricei: {taxonomy_vectors.shape}")
        return taxonomy_vectors
    
    def compute_similarity_to_taxonomy(self, company_vectors: np.ndarray, 
                                      taxonomy_vectors: np.ndarray) -> np.ndarray:
        """
        Calculează similitudinea între vectorii companiei și etichetele taxonomiei.
        
        Args:
            company_vectors: Reprezentările TF-IDF ale companiilor
            taxonomy_vectors: Reprezentările TF-IDF ale etichetelor taxonomiei
            
        Returns:
            Matrice de similitudine dintre companii și etichete
        """
        logger.info("Calculul similitudinii între companii și etichete")
        
        # Calculăm similitudinea cosinus între companii și etichete
        similarity_matrix = cosine_similarity(company_vectors, taxonomy_vectors)
        logger.info(f"Matricea de similitudine are forma: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def get_top_taxonomy_matches(self, similarity_matrix: np.ndarray, 
                               taxonomy_df: pd.DataFrame,
                               top_k: int = 3, 
                               threshold: float = 0.1) -> List[List[Dict]]:
        """
        Obține cele mai bune potriviri din taxonomie pentru fiecare companie.
        
        Args:
            similarity_matrix: Matricea de similitudine între companii și etichete
            taxonomy_df: DataFrame cu taxonomia
            top_k: Numărul maxim de etichete de returnat pentru fiecare companie
            threshold: Pragul minim de similitudine pentru a considera o potrivire
            
        Returns:
            Listă de liste cu dicționare conținând etichetele potrivite și scorurile
        """
        logger.info(f"Obținere top {top_k} potriviri pentru fiecare companie (prag: {threshold})")
        
        all_matches = []
        
        # Pentru fiecare companie
        for i in range(similarity_matrix.shape[0]):
            company_similarities = similarity_matrix[i]
            
            # Obținem indicii sortați ai celor mai similare etichete
            top_indices = company_similarities.argsort()[-top_k:][::-1]
            
            # Filtrăm pe baza pragului
            company_matches = []
            for idx in top_indices:
                score = company_similarities[idx]
                if score >= threshold:
                    company_matches.append({
                        'label': taxonomy_df.iloc[idx]['label'],
                        'score': float(score)
                    })
            
            all_matches.append(company_matches)
        
        return all_matches
    
    def assign_insurance_labels(self, df: pd.DataFrame, 
                               taxonomy_df: pd.DataFrame,
                               top_k: int = 3, 
                               threshold: float = 0.1,
                               company_text_column: str = 'combined_features',
                               taxonomy_column: str = 'cleaned_label') -> pd.DataFrame:
        """
        Asignează etichete de asigurări companiilor pe baza similitudinii TF-IDF.
        
        Args:
            df: DataFrame cu datele companiei preprocesate
            taxonomy_df: DataFrame cu taxonomia de asigurări
            top_k: Numărul maxim de etichete de asignat unei companii
            threshold: Pragul minim de similitudine pentru a asigna o etichetă
            company_text_column: Coloana din df conținând textul pentru comparare
            taxonomy_column: Coloana din taxonomy_df conținând textul pentru comparare
            
        Returns:
            DataFrame original cu o coloană nouă 'insurance_label' adăugată
        """
        logger.info(f"Asignare etichete de asigurări pentru {len(df)} companii")
        
        # Antrenăm vectorizorul pe datele combinate dacă nu a fost deja antrenat
        if not hasattr(self, 'vocabulary_size'):
            self.fit_vectorizer(df, taxonomy_df, company_text_column, taxonomy_column)
        
        # Transformă datele companiei și taxonomiei în vectori TF-IDF
        company_vectors = self.transform_company_data(df, company_text_column)
        taxonomy_vectors = self.transform_taxonomy(taxonomy_df, taxonomy_column)
        
        # Calculează matricea de similitudine
        similarity_matrix = self.compute_similarity_to_taxonomy(company_vectors, taxonomy_vectors)
        
        # Obține cele mai bune potriviri pentru fiecare companie
        matches = self.get_top_taxonomy_matches(
            similarity_matrix, 
            taxonomy_df, 
            top_k=top_k, 
            threshold=threshold
        )
        
        # Adaugă etichetele la dataframe
        result_df = df.copy()
        
        # Creăm o coloană cu liste de etichete și o coloană cu liste de scoruri
        result_df['insurance_labels'] = [
            [match['label'] for match in company_matches] 
            for company_matches in matches
        ]
        
        result_df['insurance_label_scores'] = [
            [match['score'] for match in company_matches] 
            for company_matches in matches
        ]
        
        # Creăm o coloană 'insurance_label' conform cerințelor, cu etichetele separate prin virgulă
        result_df['insurance_label'] = [
            ', '.join(labels) if labels else 'Unclassified'
            for labels in result_df['insurance_labels']
        ]
        
        logger.info("Asignare etichete completă")
        
        return result_df
    
    def save_models(self, filename_prefix: str = 'tfidf_model') -> None:
        """
        Salvează modelul TF-IDF antrenat.
        
        Args:
            filename_prefix: Prefixul numelui fișierului pentru salvare
        """
        logger.info(f"Salvare model TF-IDF în directorul {self.models_path}")
        
        if hasattr(self, 'vectorizer') and hasattr(self.vectorizer, 'vocabulary_'):
            save_path = os.path.join(self.models_path, f"{filename_prefix}.joblib")
            joblib.dump(self.vectorizer, save_path)
            logger.info(f"Model salvat la {save_path}")
        else:
            logger.warning("Nu s-a găsit model antrenat pentru salvare")
    
    def load_models(self, filepath: str) -> bool:
        """
        Încarcă modelul TF-IDF pre-antrenat.
        
        Args:
            filepath: Calea către fișierul cu modelul salvat
            
        Returns:
            Bool indicând dacă încărcarea a avut succes
        """
        logger.info(f"Încărcare model TF-IDF din {filepath}")
        
        try:
            self.vectorizer = joblib.load(filepath)
            
            if hasattr(self.vectorizer, 'vocabulary_'):
                self.vocabulary_size = len(self.vectorizer.vocabulary_)
                logger.info(f"Model încărcat cu succes: vocabular cu {self.vocabulary_size} termeni")
                return True
            else:
                logger.error("Modelul încărcat nu are un vocabular valid")
                return False
        except Exception as e:
            logger.error(f"Eroare la încărcarea modelului: {e}")
            return False


# Exemplu de utilizare
if __name__ == "__main__":
    # Acesta este doar un exemplu - în implementarea reală vei folosi datele tale
    
    from src.preprocessing.preprocessing import DataPreprocessor
    
    # Inițializăm preprocesorul de date
    data_preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Încărcăm și preprocesăm datele
    companies_df, taxonomy_df = data_preprocessor.process_company_and_taxonomy(
        company_file="companies.csv",
        taxonomy_file="insurance_taxonomy.csv"
    )
    
    # Inițializăm procesorul TF-IDF
    tfidf_processor = TFIDFProcessor(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 3),
        models_path='models/'
    )
    
    # Asignăm etichete de asigurări companiilor
    # (fit_vectorizer va fi apelat automat în assign_insurance_labels)
    classified_companies = tfidf_processor.assign_insurance_labels(
        companies_df,
        taxonomy_df,
        top_k=3,
        threshold=0.05,
        company_text_column='combined_features',
        taxonomy_column='cleaned_label'
    )
    
    # Salvăm rezultatele
    classified_companies.to_csv('data/processed/classified_companies.csv', index=False)
    
    # Salvăm modelele pentru utilizare ulterioară
    tfidf_processor.save_models()