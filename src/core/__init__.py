"""
Core engine modules for Stock Markov Analysis.

This package contains the fundamental components of the Markov chain
stock analysis system including data management, models, and simulations.
"""

from .data.fetcher import StockDataFetcher
from .data.preprocessor import DataPreprocessor
from .models.markov import MarkovChain, HigherOrderMarkov
from .simulation.simulator import MonteCarloSimulator
from .metrics.performance import PerformanceMetrics

__all__ = [
    'StockDataFetcher',
    'DataPreprocessor',
    'MarkovChain',
    'HigherOrderMarkov',
    'MonteCarloSimulator',
    'PerformanceMetrics',
]

__version__ = '2.0.0'
