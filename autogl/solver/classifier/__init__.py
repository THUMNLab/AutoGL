"""
Auto classifier for classification problems.
"""

from .base import BaseClassifier
from .graph_classifier import AutoGraphClassifier
from .node_classifier import AutoNodeClassifier
from .link_predictor import AutoLinkPredictor
from autogl.backend import DependentBackend
if DependentBackend.is_dgl():
    from .hetero import AutoHeteroNodeClassifier

__all__ = [
    "BaseClassifier",
    "AutoGraphClassifier",
    "AutoNodeClassifier",
    "AutoLinkPredictor",
]

if DependentBackend.is_dgl():
    __all__.extend(['AutoHeteroNodeClassifier'])