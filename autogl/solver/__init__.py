"""
Auto solver for various graph tasks
"""

from .classifier import AutoGraphClassifier, AutoNodeClassifier
from .utils import Leaderboard

__all__ = ["AutoNodeClassifier", "AutoGraphClassifier", "Leaderboard"]
