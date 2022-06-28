from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseAutoModel
from .topkpool import AutoTopkpool

# from .graph_sage import AutoSAGE
from .graphsage import AutoSAGE
from .graph_saint import GraphSAINTAggregationModel
from .gcn import AutoGCN
from .gat import AutoGAT
from .gin import AutoGIN
from .gin_gat_glf import AutoGATPooling

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
    "AutoGATPooling"
]
