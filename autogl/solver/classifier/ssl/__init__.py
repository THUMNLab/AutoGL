from ....backend import DependentBackend

__all__ = []

if DependentBackend.is_pyg():
    from .ssl_graph_classifier import SSLGraphClassifier
    __all__.extend(["SSLGraphClassifier"])
