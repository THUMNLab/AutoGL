"""
Auto solver for various graph tasks
"""

from autogl.backend import DependentBackend
from .classifier import AutoGraphClassifier, AutoNodeClassifier, AutoLinkPredictor

if DependentBackend.is_dgl():
    from .classifier import AutoHeteroNodeClassifier

from .utils import LeaderBoard

__all__ = [
    "AutoNodeClassifier",
    "AutoGraphClassifier",
    "AutoLinkPredictor",
    "LeaderBoard",
]

if DependentBackend.is_dgl():
    __all__.append("AutoHeteroNodeClassifier")