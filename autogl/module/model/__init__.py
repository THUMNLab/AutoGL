import importlib
import sys
from ...backend import DependentBackend
from . import _utils

from .decoders import (
    BaseDecoderMaintainer,
    DecoderUniversalRegistry,
    LogSoftmaxDecoderMaintainer,
    DotProductLinkPredictionDecoderMaintainer
)

from .encoders import (
    BaseEncoderMaintainer,
    AutoHomogeneousEncoderMaintainer,
    EncoderUniversalRegistry,
    GCNEncoderMaintainer,
    GATEncoderMaintainer,
    GINEncoderMaintainer,
    SAGEEncoderMaintainer
)

if DependentBackend.is_dgl():
    from .decoders import (
        TopKDecoderMaintainer,
        JKSumPoolDecoderMaintainer
    )
else:
    from .decoders import (
        DiffPoolDecoderMaintainer,
        SumPoolMLPDecoderMaintainer
    )

# load corresponding backend model of subclass
def _load_subclass_backend(backend):
    sub_module = importlib.import_module(f'.{backend.get_backend_name()}', __name__)
    this = sys.modules[__name__]
    for api, obj in sub_module.__dict__.items():
        setattr(this, api, obj)

_load_subclass_backend(DependentBackend)

__all__.extend([
    "BaseDecoderMaintainer",
    "DecoderUniversalRegistry",
    "LogSoftmaxDecoderMaintainer",
    "DotProductLinkPredictionDecoderMaintainer",
    "BaseEncoderMaintainer",
    "AutoHomogeneousEncoderMaintainer",
    "EncoderUniversalRegistry",
    "GCNEncoderMaintainer",
    "GATEncoderMaintainer",
    "GINEncoderMaintainer",
    "SAGEEncoderMaintainer"
])

if DependentBackend.is_dgl():
    __all__.extend([
        "TopKDecoderMaintainer",
        "JKSumPoolDecoderMaintainer",

    ])
else:
    __all__.extend([
        "DiffPoolDecoderMaintainer",
        "SumPoolMLPDecoderMaintainer"
    ])
