import autogl
from ._graph_sampler import GraphSampler, SampledSubgraph, GraphSamplerUniversalRegistry, instantiate_graph_sampler


if autogl.backend.DependentBackend.is_pyg():
    from ._pyg import (
        PyGGraphSampler, PyGHomogeneousGraphSampler, PyGSampledSubgraph,
        PyGClusterSampler, PyGNeighborSampler,
        PyGGraphSAINTNodeSampler, PyGGraphSAINTEdgeSampler, PyGGraphSAINTRandomWalkSampler
    )

    __all__ = [
        'GraphSampler',
        'SampledSubgraph',
        'GraphSamplerUniversalRegistry',
        'instantiate_graph_sampler',
        'PyGGraphSampler',
        'PyGHomogeneousGraphSampler',
        'PyGSampledSubgraph',
        'PyGClusterSampler',
        'PyGNeighborSampler',
        'PyGGraphSAINTNodeSampler',
        'PyGGraphSAINTEdgeSampler',
        'PyGGraphSAINTRandomWalkSampler'
    ]
