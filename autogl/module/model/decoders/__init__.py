from .base_decoder import BaseDecoderMaintainer
from .decoder_registry import DecoderUniversalRegistry
from autogl.backend import DependentBackend

if DependentBackend.is_pyg():
    from ._pyg import (
        LogSoftmaxDecoderMaintainer,
        AddPoolMLPDecoderMaintainer,
        DiffPoolDecoderMaintainer,
        DotProductLinkPredictonDecoderMaintainer
    )
else:
    from ._dgl import (
        LogSoftmaxDecoderMaintainer,
        JKSumPoolDecoderMaintainer,
        TopKDecoderMaintainer,
        DotProductLinkPredictionDecoderMaintainer
    )

__all__ = [
    "BaseDecoderMaintainer",
    "DecoderUniversalRegistry",
    "LogSoftmaxDecoderMaintainer",
    "JKSumPoolDecoderMaintainer",
    "TopKDecoderMaintainer",
    "DiffPoolDecoderMaintainer",
    "DotProductLinkPredictonDecoderMaintainer"
]
