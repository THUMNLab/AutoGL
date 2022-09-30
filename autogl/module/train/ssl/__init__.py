from ....backend import DependentBackend

if DependentBackend.is_pyg():
    from .graphcl import GraphCLSemisupervisedTrainer, GraphCLUnsupervisedTrainer, BaseContrastiveTrainer
    from .utils import get_view_by_name
    __all__ = [
        "GraphCLSemisupervisedTrainer",
        "GraphCLUnsupervisedTrainer",
        "BaseContrastiveTrainer",
        "get_view_by_name"
    ]
else:
    from .graphcl import BaseContrastiveTrainer
    __all__ = [
        "BaseContrastiveTrainer"
    ]
