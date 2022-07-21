from ....backend import DependentBackend

if DependentBackend.is_pyg():
    from .graphcl_semisupervised_trainer import GraphCLSemisupervisedTrainer

    __all__ = [
        "GraphCLSemisupervisedTrainer"
    ]
else:
    __all__ = []