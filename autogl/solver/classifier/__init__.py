"""
Auto classifier for classification problems.
"""

from .base import BaseClassifier
from .graph_classifier import AutoGraphClassifier
from .node_classifier import AutoNodeClassifier

__all__ = ["BaseClassifier", "AutoGraphClassifier", "AutoNodeClassifier"]
