from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseAutoModel
from .topkpool import AutoTopkpool

# from .graph_sage import AutoSAGE
from .graphsage import AutoSAGE
from .graph_saint import GraphSAINTAggregationModel
from .gcn import AutoGCN
from .gat import AutoGAT
from .gin import AutoGIN

from .robust.gnnguard import AutoGNNGuard, AutoGNNGuard_attack, GCN4GNNGuard, GCN4GNNGuard_attack

__all__ = [
    "ModelUniversalRegistry",
    "register_model",
    "BaseAutoModel",
    "AutoTopkpool",
    "AutoSAGE",
    "GraphSAINTAggregationModel",
    "AutoGCN",
    "AutoGAT",
    "AutoGIN",
    "AutoGNNGuard",
    "AutoGNNGuard_attack",
    "GCN4GNNGuard",
    "GCN4GNNGuard_attack",
]
