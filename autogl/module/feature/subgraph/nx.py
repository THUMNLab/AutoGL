from networkx.algorithms.euler import is_eulerian
from networkx.algorithms.efficiency_measures import global_efficiency
from networkx.algorithms.efficiency_measures import local_efficiency
from networkx.algorithms.distance_regular import is_distance_regular
from networkx.algorithms.components import number_connected_components
from networkx.algorithms.components import is_connected
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.clique import graph_number_of_cliques
from networkx.algorithms.clique import graph_clique_number
from networkx.algorithms.bridges import has_bridges
from networkx.algorithms.assortativity import degree_pearson_correlation_coefficient
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.approximation.clique import large_clique_size
from .base import BaseSubgraph
import torch
from .. import register_feature

NX_EXTRACTORS = []


def register_nx(cls):
    NX_EXTRACTORS.append(cls)
    register_feature(cls.__name__)(cls)
    return cls


@register_nx
class NxSubgraph(BaseSubgraph):
    def __init__(self, *args, **kwargs):
        super(NxSubgraph, self).__init__(data_t="nx")
        self._args = args
        self._kwargs = kwargs

    def extract(self, data):
        return data.G.size()

    def _transform(self, data):
        dsc = self.extract(data)
        dsc = torch.FloatTensor([[dsc]])
        data.gf = torch.cat([data.gf, dsc], dim=1)
        return data


def nxfunc(func):
    r"""A decorator for networkx subgraph transforms. You may want to use it to quickly wrap a nx subgraph feature function object.

    Examples
    --------
    @register_nx
    @nxfunc(large_clique_size)
    class NxLargeCliqueSize(NxSubgraph):pass

    """

    def decorator_func(cls):
        cls.extract = lambda s, data: func(data.G, *s._args, **s._kwargs)
        return cls

    return decorator_func


@register_nx
@nxfunc(large_clique_size)
class NxLargeCliqueSize(NxSubgraph):
    pass


@register_nx
@nxfunc(average_clustering)
class NxAverageClusteringApproximate(NxSubgraph):
    pass


@register_nx
@nxfunc(degree_assortativity_coefficient)
class NxDegreeAssortativityCoefficient(NxSubgraph):
    pass


@register_nx
@nxfunc(degree_pearson_correlation_coefficient)
class NxDegreePearsonCorrelationCoefficient(NxSubgraph):
    pass


@register_nx
@nxfunc(has_bridges)
class NxHasBridge(NxSubgraph):
    pass


@register_nx
@nxfunc(graph_clique_number)
class NxGraphCliqueNumber(NxSubgraph):
    pass


@register_nx
@nxfunc(graph_number_of_cliques)
class NxGraphNumberOfCliques(NxSubgraph):
    pass


@register_nx
@nxfunc(transitivity)
class NxTransitivity(NxSubgraph):
    pass


@register_nx
@nxfunc(average_clustering)
class NxAverageClustering(NxSubgraph):
    pass


@register_nx
@nxfunc(is_connected)
class NxIsConnected(NxSubgraph):
    pass


@register_nx
@nxfunc(number_connected_components)
class NxNumberConnectedComponents(NxSubgraph):
    pass


# from networkx.algorithms.components import is_attracting_component
# @register_nx
# @nxfunc(is_attracting_component)
# class NxIsAttractingComponent(NxSubgraph):pass

# from networkx.algorithms.components import number_attracting_components
# @register_nx
# @nxfunc(number_attracting_components)
# class NxNumberAttractingComponents(NxSubgraph):pass

# from networkx.algorithms.connectivity.connectivity import average_node_connectivity
# @register_nx
# @nxfunc(average_node_connectivity)
# class NxAverageNodeConnectivity(NxSubgraph):pass

# from networkx.algorithms.distance_measures import diameter
# @register_nx
# @nxfunc(diameter)
# class NxDiameter(NxSubgraph):pass

# from networkx.algorithms.distance_measures import radius
# @register_nx
# @nxfunc(radius)
# class NxRadius(NxSubgraph):pass


@register_nx
@nxfunc(is_distance_regular)
class NxIsDistanceRegular(NxSubgraph):
    pass


@register_nx
@nxfunc(local_efficiency)
class NxLocalEfficiency(NxSubgraph):
    pass


@register_nx
@nxfunc(global_efficiency)
class NxGlobalEfficiency(NxSubgraph):
    pass


@register_nx
@nxfunc(is_eulerian)
class NxIsEulerian(NxSubgraph):
    pass


# till algorithms.flows
