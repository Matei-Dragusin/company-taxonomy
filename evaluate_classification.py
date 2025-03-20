#!/usr/bin/env python
"""
Script pentru evaluarea performanței clasificării companiilor
și analiza rezultatelor.
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

# Adăugăm directorul rădăcină al proiectului în path pentru a permite importuri
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurăm logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_classification_results(file_path: str) -> pd.DataFrame:
    """
    Încarcă rezultatele clasificării din fișierul CSV.
    
    Args:
        file_path: Calea către fișierul cu rezultate
        
    Returns:
        DataFrame cu rezultatele clasificării
    """
    logger.info(f"Încărcare rezultate clasificare din {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Fișierul {file_path} nu există!")
        return None
    
    try:
        results_df = pd.read_csv(file_path)
        logger.info(f"Încărcate {len(results_df)} înregistrări din fișierul de rezultate")
        return results_df
    except Exception as e:
        logger.error(f"Eroare la încărcarea fișierului de rezultate: {e}")
        return None

def analyze_label_distribution(results_df: pd.DataFrame) -> Dict:
    """
    Analizează distribuția etichetelor asignate.
    
    Args:
        results_df: DataFrame cu rezultatele clasificării
        
    Returns:
        Dicționar cu statistici despre distribuția etichetelor
    """
    logger.info("Analizarea distribuției etichetelor")
    
    # Convertim coloanele de liste reprezentate ca stringuri în liste Python
    if 'insurance_labels' not in results_df.columns and 'insurance_label' in results_df.columns:
        # Dacă avem doar insurance_label, o separăm în listă
        results_df['insurance_labels'] = results_df['insurance_label'].str.split(', ')
    
    # Verificăm dacă avem acces la listele de etichete
    if 'insurance_labels' not in results_df.columns:
        logger.error("Coloana 'insurance_labels' nu există în rezultate!")
        return {}
    
    # Numără toate etichetele
    all_labels = []
    for labels in results_df['insurance_labels']:
        if isinstance(labels, list):
            all_labels.extend(labels)
        elif isinstance(labels, str):
            if labels.startswith('[') and labels.endswith(']'):
                # Convertim string reprezentarea unei liste în listă Python
                try:
                    parsed_labels = eval(labels)
                    if isinstance(parsed_labels, list):
                        all_labels.extend(parsed_labels)
                except:
                    pass
            else:
                # Posibil o listă separată prin virgule
                label_items = [item.strip() for item in labels.split(',') if item.strip()]
                all_labels.extend(label_items)
    
    label_counts = Counter(all_labels)
    
    # Companii per etichetă
    label_company_counts = {}
    for label in set(all_labels):
        companies_with_label = 0
        for labels in results_df['insurance_labels']:
            if isinstance(labels, list) and label in labels:
                companies_with_label += 1
            elif isinstance(labels, str):
                if label in labels:  # Verificare simplă
                    companies_with_label += 1
        label_company_counts[label] = companies_with_label
    
    # Statistici de distribuție a etichetelor
    total_companies = len(results_df)
    total_label_assignments = len(all_labels)
    unique_labels = len(label_counts)
    avg_labels_per_company = total_label_assignments / total_companies if total_companies > 0 else 0
    
    # Top etichete și etichete mai puțin frecvente
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
    Analizează scorurile de similitudine pentru etichetele asignate.
    
    Args:
        results_df: DataFrame cu rezultatele clasificării
        
    Returns:
        Dicționar cu statistici despre scorurile de similitudine
    """
    logger.info("Analizarea scorurilor de similitudine")
    
    # Verificăm dacă avem acces la scorurile de similitudine
    if 'insurance_label_scores' not in results_df.columns:
        logger.error("Coloana 'insurance_label_scores' nu există în rezultate!")
        return {}
    
    # Colectăm toate scorurile de similitudine
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
        logger.warning("Nu s-au găsit scoruri de similitudine pentru analiză")
        return {}
    
    # Calculăm statistici despre scoruri
    avg_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    # Distribuția scorurilor
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
    Analizează acoperirea companiilor - câte companii au primit etichete.
    
    Args:
        results_df: DataFrame cu rezultatele clasificării
        
    Returns:
        Dicționar cu statistici despre acoperirea companiilor
    """
    logger.info("Analizarea acoperirii companiilor")
    
    total_companies = len(results_df)
    
    # Verificăm câte companii au primit cel puțin o etichetă
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
    
    # Distribuția numărului de etichete per companie
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
    Generează grafice pentru distribuția etichetelor.
    
    Args:
        distribution_stats: Statisticile distribuției etichetelor
        output_dir: Directorul unde vor fi salvate graficele
    """
    logger.info("Generarea graficelor pentru distribuția etichetelor")
    
    # Creăm directorul de ieșire dacă nu există
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plotăm distribuția celor mai frecvente etichete
    plt.figure(figsize=(12, 6))
    
    if 'top_labels' in distribution_stats and distribution_stats['top_labels']:
        labels = [label for label, count in distribution_stats['top_labels']]
        counts = [count for label, count in distribution_stats['top_labels']]
        
        # Folosim range pentru x și adaugăm etichetele manual
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.title('Top 10 Cele Mai Frecvente Etichete')
        plt.xlabel('Etichetă')
        plt.ylabel('Număr de Asignări')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'top_labels.png'))
    
    # Plotăm distribuția numărului de etichete per companie
    plt.figure(figsize=(10, 6))
    
    label_counts = distribution_stats.get('label_count_distribution', {})
    if label_counts:
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Convertim etichetele la string pentru siguranță
        str_labels = [str(label) for label in labels]
        
        # Folosim range pentru x și adaugăm etichetele manual
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, str_labels)
        plt.title('Distribuția Numărului de Etichete per Companie')
        plt.xlabel('Număr de Etichete')
        plt.ylabel('Număr de Companii')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'label_count_distribution.png'))
    
    # Plotăm distribuția scorurilor de similitudine
    plt.figure(figsize=(10, 6))
    
    score_ranges = distribution_stats.get('score_ranges', {})
    if score_ranges:
        ranges = list(score_ranges.keys())
        counts = list(score_ranges.values())
        
        # Folosim range pentru x și adaugăm etichetele manual
        x_pos = np.arange(len(ranges))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, ranges, rotation=45)
        plt.title('Distribuția Scorurilor de Similitudine')
        plt.xlabel('Interval de Scor')
        plt.ylabel('Număr de Asignări')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'similarity_score_distribution.png'))
    
    logger.info(f"Graficele au fost salvate în directorul {output_dir}")

def generate_evaluation_report(results_df: pd.DataFrame, output_file: str = 'results/evaluation_report.txt'):
    """
    Generează un raport de evaluare detaliat.
    
    Args:
        results_df: DataFrame cu rezultatele clasificării
        output_file: Fișierul de ieșire pentru raport
    """
    logger.info("Generarea raportului de evaluare")
    
    # Creăm directorul pentru raport dacă nu există
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Colectăm toate statisticile
    label_stats = analyze_label_distribution(results_df)
    score_stats = analyze_similarity_scores(results_df)
    coverage_stats = analyze_company_coverage(results_df)
    
    # Generăm graficele
    plot_label_distribution({**label_stats, **score_stats, **coverage_stats})
    
    # Scriem raportul
    with open(output_file, 'w') as f:
        f.write("=== RAPORT DE EVALUARE A CLASIFICĂRII ===\n\n")
        
        f.write("== STATISTICI GENERALE ==\n")
        f.write(f"Număr total de companii: {coverage_stats['total_companies']}\n")
        f.write(f"Companii cu etichete asignate: {coverage_stats['companies_with_labels']} ({coverage_stats['coverage_percentage']:.2f}%)\n")
        f.write(f"Companii fără etichete: {coverage_stats['companies_without_labels']}\n")
        f.write(f"Număr mediu de etichete per companie: {label_stats['avg_labels_per_company']:.2f}\n\n")
        
        f.write("== DISTRIBUȚIA NUMĂRULUI DE ETICHETE PER COMPANIE ==\n")
        for label_count, num_companies in coverage_stats['label_count_distribution'].items():
            f.write(f"{label_count} etichete: {num_companies} companii\n")
        f.write("\n")
        
        f.write("== STATISTICI ETICHETE ==\n")
        f.write(f"Număr total de asignări de etichete: {label_stats['total_label_assignments']}\n")
        f.write(f"Număr de etichete unice folosite: {label_stats['unique_labels']}\n\n")
        
        f.write("== TOP 10 CELE MAI FRECVENTE ETICHETE ==\n")
        for label, count in label_stats['top_labels']:
            f.write(f"{label}: {count} asignări\n")
        f.write("\n")
        
        f.write("== SCORURI DE SIMILITUDINE ==\n")
        f.write(f"Scor mediu de similitudine: {score_stats['avg_score']:.4f}\n")
        f.write(f"Scor median: {score_stats['median_score']:.4f}\n")
        f.write(f"Scor minim: {score_stats['min_score']:.4f}\n")
        f.write(f"Scor maxim: {score_stats['max_score']:.4f}\n\n")
        
        f.write("== DISTRIBUȚIA SCORURILOR DE SIMILITUDINE ==\n")
        for range_str, count in score_stats['score_ranges'].items():
            f.write(f"Interval {range_str}: {count} asignări\n")
        
        f.write("\n=== SFÂRȘIT RAPORT ===\n")
    
    logger.info(f"Raportul de evaluare a fost generat la {output_file}")
    return {**label_stats, **score_stats, **coverage_stats}

def main():
    """Funcție principală pentru rularea evaluării"""
    parser = argparse.ArgumentParser(description='Evaluarea performanței clasificării companiilor')
    
    parser.add_argument('--input-file', type=str, default='data/processed/classified_companies.csv',
                        help='Calea către fișierul cu rezultatele clasificării (implicit: data/processed/classified_companies.csv)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directorul pentru ieșire (implicit: results)')
    
    args = parser.parse_args()
    
    logger.info("Pornirea evaluării performanței clasificării")
    
    # Încărcăm rezultatele clasificării
    results_df = load_classification_results(args.input_file)
    
    if results_df is None:
        logger.error("Nu s-au putut încărca rezultatele clasificării. Programul se închide.")
        return
    
    # Creăm directorul de ieșire dacă nu există
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"S-a creat directorul de ieșire: {args.output_dir}")
    
    # Generăm raportul de evaluare
    output_file = os.path.join(args.output_dir, 'evaluation_report.txt')
    evaluation_stats = generate_evaluation_report(results_df, output_file)
    
    # Afișăm câteva statistici cheie
    print("\n=== STATISTICI CHEIE ===")
    print(f"Număr total de companii: {evaluation_stats['total_companies']}")
    print(f"Acoperire: {evaluation_stats['coverage_percentage']:.2f}% companii au primit etichete")
    print(f"Număr mediu de etichete per companie: {evaluation_stats['avg_labels_per_company']:.2f}")
    print(f"Scor mediu de similitudine: {evaluation_stats['avg_score']:.4f}")
    print(f"Etichetă cea mai frecventă: {evaluation_stats['top_labels'][0][0]} ({evaluation_stats['top_labels'][0][1]} asignări)")
    print(f"\nRaportul complet și graficele sunt disponibile în: {args.output_dir}")

if __name__ == "__main__":
    main()