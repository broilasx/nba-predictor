"""
NBA Predictor Data Package

This package handles all data-related functionality:
- Data collection from NBA API
- Data processing and feature engineering
"""

from .collector import fetch_game_data
from .processor import process_data, add_features, encode_features, prepare_features

__all__ = [
    'fetch_game_data',
    'process_data',
    'add_features',
    'encode_features',
    'prepare_features'
]