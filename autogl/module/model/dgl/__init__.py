from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseModel
from .topkpool import AutoTopkpool


from .graph_saint import GraphSAINTAggregationModel
from .gcn import GCN, AutoGCN
from .graphsage import GraphSAGE, AutoSAGE
from .gat import GAT,AutoGAT
from .gin import AutoGIN

__all__ = [
    "ModelUniversalRegistry",
    "register_model",
    "BaseModel",
    "AutoTopkpool",
    "GraphSAINTAggregationModel",
    "GCN",
    "AutoGCN",
    "GraphSAGE",
    "AutoSAGE",
    "GAT",
    "AutoGAT",
    "AutoGIN"
]
