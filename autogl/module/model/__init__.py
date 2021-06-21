from ._model_registry import MODEL_DICT, ModelUniversalRegistry, register_model
from .base import BaseModel
from .topkpool import AutoTopkpool

# from .graph_sage import AutoSAGE
from .graphsage import AutoSAGE
from .gcn import AutoGCN
from .gat import AutoGAT
from .gin import AutoGIN
