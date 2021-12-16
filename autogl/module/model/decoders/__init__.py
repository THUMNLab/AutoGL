from .base_decoder import BaseAutoDecoderMaintainer
from .decoder_registry import DecoderUniversalRegistry
from autogl.backend import DependentBackend

if DependentBackend.is_pyg():
    from ._pyg import (
        LogSoftmaxDecoderMaintainer,
        GINDecoderMaintainer,
        DiffPoolDecoderMaintainer,
        DotProductLinkPredictonDecoderMaintainer
    )
else:
    from ._dgl import (
        LogSoftmaxDecoderMaintainer,
        GINDecoderMaintainer,
        TopKDecoderMaintainer,
        DotProductLinkPredictonDecoderMaintainer
    )

__all__ = [
    "BaseAutoDecoderMaintainer",
    "DecoderUniversalRegistry",
    "LogSoftmaxDecoderMaintainer",
    "GINDecoderMaintainer",
    "TopKDecoderMaintainer",
    "DiffPoolDecoderMaintainer",
    "DotProductLinkPredictonDecoderMaintainer"
]
