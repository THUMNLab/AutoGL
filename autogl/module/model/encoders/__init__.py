from .base_encoder import BaseEncoderMaintainer, AutoHomogeneousEncoderMaintainer
from .encoder_registry import EncoderUniversalRegistry
from autogl.backend import DependentBackend

if DependentBackend.is_pyg():
    from ._pyg import (
        GCNEncoderMaintainer,
        GATEncoderMaintainer,
        GINEncoderMaintainer,
        SAGEEncoderMaintainer
    )
else:
    from ._dgl import (
        GCNEncoderMaintainer,
        GATEncoderMaintainer,
        GINEncoderMaintainer,
        SAGEEncoderMaintainer,
        AutoTopKMaintainer
    )

__all__ = [
    "BaseEncoderMaintainer",
    "EncoderUniversalRegistry",
    "AutoHomogeneousEncoderMaintainer",
    "GCNEncoderMaintainer",
    "GATEncoderMaintainer",
    "GINEncoderMaintainer",
    "SAGEEncoderMaintainer",
]

if DependentBackend.is_dgl():
    __all__.append("AutoTopKMaintainer")
