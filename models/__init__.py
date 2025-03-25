"""
NBA Predictor Models Package

This package contains all model-related functionality:
- Model training and evaluation
- Making predictions with trained models
"""

from .trainer import train_model
from .predictor import predict_upcoming_games

__all__ = ['train_model', 'predict_upcoming_games']