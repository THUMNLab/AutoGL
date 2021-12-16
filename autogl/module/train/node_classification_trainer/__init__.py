from ....backend import DependentBackend

if DependentBackend.is_pyg():
    from .node_classification_sampled_trainer import *
else:
    NodeClassificationGraphSAINTTrainer = None
    NodeClassificationLayerDependentImportanceSamplingTrainer = None
    NodeClassificationNeighborSamplingTrainer = None
    pass





