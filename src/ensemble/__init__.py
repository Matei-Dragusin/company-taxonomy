"""
Ensemble classification module for insurance taxonomy.

This module contains classifiers that combine multiple approaches
to classify companies into insurance taxonomy categories:

1. EnsembleClassifier - Basic ensemble classifier combining TF-IDF, 
   WordNet similarity, and keyword matching.
   
2. OptimizedEnsembleClassifier - Performance-optimized version for 
   large datasets with batching and caching mechanisms.
"""

from src.ensemble.ensemble_classifier import EnsembleClassifier
from ensemble.fixed_optimized_ensemble import OptimizedEnsembleClassifier

__all__ = ['EnsembleClassifier', 'OptimizedEnsembleClassifier']