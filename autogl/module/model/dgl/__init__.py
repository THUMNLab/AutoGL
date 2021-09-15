from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseModel
from .topkpool import AutoTopkpool

# from .graph_sage import AutoSAGE
from .graph_saint import GraphSAINTAggregationModel
from .gcn_dgl import GCN,AutoGCN
from .graphsage_dgl import GraphSAGE
from .gat_dgl import GAT

__all__ = [
    "ModelUniversalRegistry",
    "register_model",
    "BaseModel",
    "AutoTopkpool",
    "GraphSAINTAggregationModel",
    "GCN",
    "AutoGCN",
    "GraphSAGE",
    "GAT"
]
