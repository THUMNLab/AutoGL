from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseModel
from .topkpool import AutoTopkpool

# from .graph_sage import AutoSAGE
from .graphsage import AutoSAGE
from .graph_saint import GraphSAINTAggregationModel
from .gcn import AutoGCN
from .gat import AutoGAT
from .gin import AutoGIN
from .gin_dgl import GIN
from .gcn_dgl import GCN
from .graphsage_dgl import GraphSAGE
from .gat_dgl import GAT

__all__ = [
    "ModelUniversalRegistry",
    "register_model",
    "BaseModel",
    "AutoTopkpool",
    "AutoSAGE",
    "GraphSAINTAggregationModel",
    "AutoGCN",
    "AutoGAT",
    "AutoGIN",
    "GIN",
    "GCN",
    "GraphSAGE",
    "GAT"
]
