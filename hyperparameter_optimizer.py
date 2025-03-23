#!/usr/bin/env python
"""
Script pentru optimizarea automată a hiperparametrilor clasificatorului.
Utilizează Optuna pentru a găsi valorile optime pentru parametrii clasificatorului.
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

# Adaugă directorul rădăcină al proiectului în path pentru a permite importuri
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importă modulele necesare
from src.preprocessing.preprocessing import DataPreprocessor
from src.feature_engineering.tfidf_processor import TFIDFProcessor
from src.ensemble.ensemble_classifier import EnsembleClassifier

# Configurează logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'parameter_optimization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Asigură existența directoarelor necesare
def ensure_directories_exist():
    """Asigură existența directoarelor necesare"""
    dirs = ['data/raw', 'data/processed', 'models', 'results', 'logs', 'optimization_results']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"S-a creat directorul: {dir_path}")

# Încarcă și preprocesează datele
def load_and_preprocess_data(company_file: str, taxonomy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Încarcă și preprocesează datele companiilor și taxonomiei.
    
    Args:
        company_file: Numele fișierului cu companii
        taxonomy_file: Numele fișierului cu taxonomie
        
    Returns:
        Tuple cu DataFrame-urile preprocesate pentru companii și taxonomie
    """
    logger.info(f"Încărcare și preprocesare date din {company_file} și {taxonomy_file}")
    
    # Verifică dacă există date preprocesate
    processed_companies_path = os.path.join('data/processed', 'processed_companies.csv')
    processed_taxonomy_path = os.path.join('data/processed', 'processed_taxonomy.csv')
    
    if os.path.exists(processed_companies_path) and os.path.exists(processed_taxonomy_path):
        logger.info("Se încarcă datele preprocesate existente...")
        companies_df = pd.read_csv(processed_companies_path)
        taxonomy_df = pd.read_csv(processed_taxonomy_path)
        return companies_df, taxonomy_df
    
    # Inițializează preprocesorul
    preprocessor = DataPreprocessor(
        raw_data_path='data/raw/',
        processed_data_path='data/processed/'
    )
    
    # Procesează datele
    companies_df, taxonomy_df = preprocessor.process_company_and_taxonomy(
        company_file=company_file,
        taxonomy_file=taxonomy_file,
        output_company_file="processed_companies.csv",
        output_taxonomy_file="processed_taxonomy.csv"
    )
    
    return companies_df, taxonomy_df

# Funcție pentru evaluarea unui model cu un set specific de parametri
def evaluate_parameters(
    companies_df: pd.DataFrame, 
    taxonomy_df: pd.DataFrame,
    validation_set: pd.DataFrame,
    params: Dict
) -> float:
    """
    Evaluează un set de parametri pe datele de validare.
    
    Args:
        companies_df: DataFrame cu datele companiilor
        taxonomy_df: DataFrame cu taxonomia
        validation_set: DataFrame cu set de validare anotat
        params: Dicționar cu parametri de testat
        
    Returns:
        Scor de performanță (mai mare este mai bun)
    """
    # Extragem parametrii din dicționar
    top_k = params.get('top_k', 5)
    threshold = params.get('threshold', 0.08)
    batch_size = params.get('batch_size', 100)
    tfidf_weight = params.get('tfidf_weight', 0.5)
    wordnet_weight = params.get('wordnet_weight', 0.25)
    keyword_weight = params.get('keyword_weight', 0.25)
    
    # Creăm clasificatorul cu parametrii specificați
    ensemble_classifier = EnsembleClassifier(
        models_path='models/temp/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        optimizer_mode=True
    )
    
    # Rulăm clasificarea pe un subset mic pentru eficiență
    # (putem folosi un eșantion mai mic pentru optimizare)
    sample_size = min(200, len(companies_df))
    sample_companies = companies_df.sample(n=sample_size, random_state=42)
    
    try:
        # Clasificăm companiile
        classified_companies = ensemble_classifier.ensemble_classify(
            sample_companies,
            taxonomy_df,
            top_k=top_k,
            threshold=threshold,
            company_text_column='combined_features',
            batch_size=batch_size
        )
        
        # Calculăm metricile de performanță
        # Dacă avem date anotat manual, putem calcula precizia, recall, F1
        # Dacă nu, putem folosi metrici euristice:
        
        # 1. Acoperirea (procentul de companii cu cel puțin o etichetă)
        coverage = (classified_companies['insurance_label'] != 'Unclassified').mean()
        
        # 2. Scorul mediu de încredere
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
        
        # 3. Diversitatea etichetelor (numărul de etichete unice folosite)
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
        
        # Combinăm metricile într-un scor final
        # Putem ajusta ponderile în funcție de importanța fiecărei metrici
        final_score = (0.4 * coverage) + (0.4 * mean_confidence) + (0.2 * label_diversity)
        
        # Dacă avem date de validare anotat manual, folosim F1 score
        if validation_set is not None and len(validation_set) > 0:
            # Implementează logica de evaluare pe date anotat
            # ...
            pass
        
        return final_score
        
    except Exception as e:
        logger.error(f"Eroare la evaluarea parametrilor: {e}")
        return 0.0  # În caz de eroare, returnăm un scor minim

# Funcție obiectiv pentru Optuna
def objective(trial, companies_df, taxonomy_df, validation_set=None):
    """
    Funcție obiectiv pentru optimizarea Optuna.
    
    Args:
        trial: Obiect trial Optuna
        companies_df: DataFrame cu datele companiilor
        taxonomy_df: DataFrame cu taxonomia
        validation_set: Date de validare optional
        
    Returns:
        Scor de performanță
    """
    # Definim spațiul de căutare a parametrilor
    params = {
        'top_k': trial.suggest_int('top_k', 1, 10),
        'threshold': trial.suggest_float('threshold', 0.01, 0.3),
        'batch_size': trial.suggest_categorical('batch_size', [50, 100, 200]),
    }
    
    # Folosim o abordare diferită pentru a asigura că ponderile se însumează la 1
    # Sugeram ponderi relative, apoi le normalizăm
    tfidf_relative = trial.suggest_float('tfidf_relative', 0.1, 1.0)
    wordnet_relative = trial.suggest_float('wordnet_relative', 0.1, 1.0)
    keyword_relative = trial.suggest_float('keyword_relative', 0.1, 1.0)
    
    # Normalizăm ponderile ca să se însumeze la 1
    total_weight = tfidf_relative + wordnet_relative + keyword_relative
    params['tfidf_weight'] = tfidf_relative / total_weight
    params['wordnet_weight'] = wordnet_relative / total_weight
    params['keyword_weight'] = keyword_relative / total_weight
    
    # Evaluăm parametrii
    score = evaluate_parameters(companies_df, taxonomy_df, validation_set, params)
    
    return score

# Funcție pentru optimizarea hiperparametrilor
def optimize_hyperparameters(
    companies_df: pd.DataFrame, 
    taxonomy_df: pd.DataFrame,
    validation_set: Optional[pd.DataFrame] = None,
    n_trials: int = 50,
    timeout: int = 3600,  # 1 oră implicit
    study_name: str = "insurance_taxonomy_optimization"
) -> Dict:
    """
    Optimizează hiperparametrii folosind Optuna.
    
    Args:
        companies_df: DataFrame cu datele companiilor
        taxonomy_df: DataFrame cu taxonomia
        validation_set: Date de validare optional
        n_trials: Numărul de încercări pentru optimizare
        timeout: Timpul maxim (în secunde) pentru optimizare
        study_name: Numele studiului Optuna
        
    Returns:
        Dicționar cu parametrii optimizați
    """
    logger.info(f"Începere optimizare hiperparametri cu {n_trials} încercări...")
    
    # Creăm un director pentru a stoca datele studiului
    study_dir = os.path.join('optimization_results')
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    
    # Creăm un nou studiu Optuna
    db_path = f"sqlite:///{os.path.join(study_dir, f'{study_name}.db')}"
    
    # Setăm pruner și sampler pentru eficiență
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = TPESampler(seed=42)
    
    # Creăm sau încărcăm studiul
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )
    
    # Rulăm optimizarea
    start_time = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, companies_df, taxonomy_df, validation_set),
            n_trials=n_trials,
            timeout=timeout
        )
    except KeyboardInterrupt:
        logger.info("Optimizare întreruptă manual.")
    
    duration = time.time() - start_time
    logger.info(f"Optimizare finalizată în {duration:.2f} secunde.")
    
    # Obținem parametrii optimizați
    best_params = study.best_params
    best_score = study.best_value
    
    # Afișăm rezultatele
    logger.info(f"Cel mai bun scor: {best_score}")
    logger.info(f"Parametri optimizați: {best_params}")
    
    # Salvăm rezultatele într-un fișier
    results_file = os.path.join(study_dir, f"{study_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Cel mai bun scor: {best_score}\n")
        f.write(f"Parametri optimizați:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\nIstoricul încercărilor:\n")
        # Sortăm doar încercările care au o valoare validă
        valid_trials = [t for t in study.trials if t.value is not None]
        for trial in sorted(valid_trials, key=lambda t: t.value, reverse=True):
            f.write(f"  Trial {trial.number}, Score: {trial.value}\n")
            f.write(f"  Params: {trial.params}\n\n")
    
    # Salvăm și grafice pentru analiza hiperparametrilor
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_param_importances, plot_optimization_history
        
        # Istoricul optimizării
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(study_dir, f"{study_name}_history.png"))
        
        # Importanța parametrilor
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(study_dir, f"{study_name}_importance.png"))
        
    except Exception as e:
        logger.warning(f"Nu s-au putut genera graficele: {e}")
    
    return best_params

# Funcție pentru rularea clasificării cu parametri optimizați
def run_with_optimized_parameters(
    companies_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    optimized_params: Dict,
    output_file: str = "optimized_classification.csv"
) -> pd.DataFrame:
    """
    Rulează clasificarea cu parametrii optimizați.
    
    Args:
        companies_df: DataFrame cu datele companiilor
        taxonomy_df: DataFrame cu taxonomia
        optimized_params: Dicționar cu parametri optimizați
        output_file: Numele fișierului de ieșire
        
    Returns:
        DataFrame cu rezultatele clasificării
    """
    logger.info(f"Rulare clasificare cu parametri optimizați: {optimized_params}")
    
    # Extragem parametrii
    top_k = optimized_params.get('top_k', 5)
    threshold = optimized_params.get('threshold', 0.08)
    batch_size = optimized_params.get('batch_size', 100)
    tfidf_weight = optimized_params.get('tfidf_weight', 0.5)
    wordnet_weight = optimized_params.get('wordnet_weight', 0.25)
    keyword_weight = optimized_params.get('keyword_weight', 0.25)
    
    # Inițializăm clasificatorul ensemble
    ensemble_classifier = EnsembleClassifier(
        models_path='models/',
        tfidf_weight=tfidf_weight,
        wordnet_weight=wordnet_weight,
        keyword_weight=keyword_weight,
        optimizer_mode=True
    )
    
    # Clasificăm companiile
    classified_companies = ensemble_classifier.ensemble_classify(
        companies_df,
        taxonomy_df,
        top_k=top_k,
        threshold=threshold,
        company_text_column='combined_features',
        batch_size=batch_size
    )
    
    # Salvăm modelele pentru utilizare ulterioară
    ensemble_classifier.save_models(filename_prefix='optimized_ensemble_model')
    
    # Salvăm rezultatele
    output_path = os.path.join('data/processed', output_file)
    classified_companies.to_csv(output_path, index=False)
    logger.info(f"Rezultate salvate în {output_path}")
    
    # Salvăm și un fișier simplificat
    ensemble_classifier.export_description_label_csv(
        classified_companies,
        output_path=os.path.join('data/processed', 'optimized_description_label.csv'),
        description_column='description'
    )
    
    return classified_companies

# Funcție principală
def main():
    """Funcție principală pentru rularea optimizării hiperparametrilor"""
    parser = argparse.ArgumentParser(description='Optimizare hiperparametri pentru clasificarea taxonomiei')
    
    parser.add_argument('--company-file', type=str, default='companies.csv',
                        help='Numele fișierului cu companii (implicit: companies.csv)')
    
    parser.add_argument('--taxonomy-file', type=str, default='insurance_taxonomy.csv',
                        help='Numele fișierului cu taxonomie (implicit: insurance_taxonomy.csv)')
    
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Numărul de încercări pentru optimizare (implicit: 50)')
    
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timpul maxim în secunde pentru optimizare (implicit: 3600)')
    
    parser.add_argument('--study-name', type=str, default='insurance_taxonomy_optimization',
                        help='Numele studiului Optuna (implicit: insurance_taxonomy_optimization)')
    
    parser.add_argument('--run-optimized', action='store_true',
                        help='Rulează clasificarea cu parametrii optimizați după optimizare')
    
    parser.add_argument('--output-file', type=str, default='optimized_classified_companies.csv',
                        help='Numele fișierului de ieșire pentru clasificare (implicit: optimized_classified_companies.csv)')
    
    args = parser.parse_args()
    
    logger.info("Începere optimizare hiperparametri pentru clasificarea taxonomiei")
    
    # Asigurăm existența directoarelor necesare
    ensure_directories_exist()
    
    # Încărcăm și preprocesăm datele
    try:
        companies_df, taxonomy_df = load_and_preprocess_data(
            args.company_file, 
            args.taxonomy_file
        )
    except Exception as e:
        logger.error(f"Eroare la încărcarea/preprocesarea datelor: {e}")
        return
    
    # Optimizăm hiperparametrii
    try:
        optimized_params = optimize_hyperparameters(
            companies_df,
            taxonomy_df,
            validation_set=None,  # Aici poți adăuga date anotat manual dacă ai
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=args.study_name
        )
        
        logger.info(f"Parametri optimizați obținuți: {optimized_params}")
        
        # Rulăm clasificarea cu parametrii optimizați dacă este cerut
        if args.run_optimized:
            classified_companies = run_with_optimized_parameters(
                companies_df,
                taxonomy_df,
                optimized_params,
                output_file=args.output_file
            )
            
            logger.info(f"Clasificare cu parametri optimizați finalizată!")
    except Exception as e:
        logger.error(f"Eroare în timpul optimizării: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    logger.info("Optimizare hiperparametri finalizată!")

if __name__ == "__main__":
    main()