from ....backend import DependentBackend

if DependentBackend.is_pyg():
    from .node_classification_sampled_trainer import (
        NodeClassificationGraphSAINTTrainer,
        NodeClassificationLayerDependentImportanceSamplingTrainer,
        NodeClassificationNeighborSamplingTrainer
    )
    __all__ = [
        "NodeClassificationGraphSAINTTrainer",
        "NodeClassificationLayerDependentImportanceSamplingTrainer",
        "NodeClassificationNeighborSamplingTrainer"
    ]
else:
    __all__ = []
