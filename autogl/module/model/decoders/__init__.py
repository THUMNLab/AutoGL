from .base_decoder import BaseDecoderMaintainer
from .decoder_registry import DecoderUniversalRegistry
from autogl.backend import DependentBackend

if DependentBackend.is_pyg():
    from ._pyg import (
        LogSoftmaxDecoderMaintainer,
        SumPoolMLPDecoderMaintainer,
        DiffPoolDecoderMaintainer,
        DotProductLinkPredictionDecoderMaintainer
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
    "DotProductLinkPredictionDecoderMaintainer"
]

if DependentBackend.is_pyg():
    __all__.extend([
        "DiffPoolDecoderMaintainer",
        "SumPoolMLPDecoderMaintainer"
    ])
else:
    __all__.extend([
        "JKSumPoolDecoderMaintainer",
        "TopKDecoderMaintainer"
    ])
