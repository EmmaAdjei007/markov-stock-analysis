"""
Configuration management module for Stock Markov Analysis.
Provides centralized configuration loading from YAML files and environment variables.
"""

from .config_loader import Config, get_config

__all__ = ['Config', 'get_config']
