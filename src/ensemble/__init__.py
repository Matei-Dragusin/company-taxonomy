"""
Ensemble classification module for insurance taxonomy.

This module contains classifiers that combine multiple approaches
to classify companies into insurance taxonomy categories:

1. EnsembleClassifier - Basic ensemble classifier combining TF-IDF, 
   WordNet similarity, and keyword matching.
   
2. OptimizedEnsembleClassifier - Performance-optimized version for 
   large datasets with batching and caching mechanisms.
   
3. FixedOptimizedEnsembleClassifier - Fixed version of the optimized
   ensemble classifier with better error handling.
   
4. AdaptiveEnsembleClassifier - Adaptive ensemble that learns optimal
   weights for different similarity measures.
"""

from src.ensemble.ensemble_classifier import EnsembleClassifier

__all__ = [
    'EnsembleClassifier']