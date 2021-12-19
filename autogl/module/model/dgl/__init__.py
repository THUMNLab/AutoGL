from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseAutoModel
from .topkpool import AutoTopkpool


# from .graph_saint import GraphSAINTAggregationModel
from .gcn import GCN, AutoGCN
from .graphsage import GraphSAGE, AutoSAGE
from .gat import GAT,AutoGAT
from .gin import AutoGIN
from .hetero.hgt import AutoHGT
from .hetero.han import AutoHAN
from .hetero.HeteroRGCN import AutoHeteroRGCN

__all__ = [
    "ModelUniversalRegistry",
    "register_model",
    "BaseAutoModel",
    "AutoTopkpool",
    # "GraphSAINTAggregationModel",
    "GCN",
    "AutoGCN",
    "GraphSAGE",
    "AutoSAGE",
    "GAT",
    "AutoGAT",
    "AutoGIN",
    "AutoHGT",
    "AutoHAN", 
    "AutoHeteroRGCN",
]
