"""
Auto solver for various graph tasks
"""

from .classifier import AutoGraphClassifier, AutoNodeClassifier, AutoLinkPredictor
from .utils import LeaderBoard

__all__ = [
    "AutoNodeClassifier",
    "AutoGraphClassifier",
    "AutoLinkPredictor",
    "LeaderBoard",
]
