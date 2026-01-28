"""
Models package for brain tumor detection.

This package contains different deep learning models for comparing
their performance on brain tumor detection task.
"""

from .cnn import build_model as build_cnn
from .dnn import build_model as build_dnn
from .lstm import build_model as build_lstm
from .inception import build_model as build_inception
from .attention_cnn import build_model as build_attention_cnn

__all__ = [
    'build_cnn',
    'build_dnn',
    'build_lstm',
    'build_inception',
    'build_attention_cnn'
]
