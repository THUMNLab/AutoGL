"""
Auto classifier for classification problems.
"""

from .base import BaseClassifier
from .graph_classifier import AutoGraphClassifier
from .node_classifier import AutoNodeClassifier
from .link_predictor import AutoLinkPredictor

__all__ = [
    "BaseClassifier",
    "AutoGraphClassifier",
    "AutoNodeClassifier",
    "AutoLinkPredictor",
]
