from ....backend import DependentBackend

if DependentBackend.is_pyg():
    from .graphcl_semisupervised_trainer import GraphCLSemisupervisedTrainer
    from .utils import get_view_by_name

    __all__ = [
        "GraphCLSemisupervisedTrainer",
        "get_view_by_name"
    ]
else:
    __all__ = []