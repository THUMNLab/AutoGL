from networkx.algorithms.euler import is_eulerian
from networkx.algorithms.efficiency_measures import global_efficiency
from networkx.algorithms.efficiency_measures import local_efficiency
from networkx.algorithms.distance_regular import is_distance_regular
from networkx.algorithms.components import number_connected_components
from networkx.algorithms.components import is_connected
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.clique import graph_number_of_cliques
from networkx.algorithms.clique import graph_clique_number
from networkx.algorithms.bridges import has_bridges
from networkx.algorithms.assortativity import degree_pearson_correlation_coefficient
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.approximation.clique import large_clique_size
import netlsd
from .base import BaseGraph
import numpy as np
import torch
from functools import wraps
from .. import register_feature

NX_EXTRACTORS = []


def register_nx(cls):
    NX_EXTRACTORS.append(cls)
    register_feature(cls.__name__)(cls)
    return cls


@register_nx
class NxGraph(BaseGraph):
    def __init__(self, *args, **kwargs):
        super(NxGraph, self).__init__(data_t="nx")
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
    r"""A decorator for networkx Graph transforms. You may want to use it to quickly wrap a nx Graph feature function object.

    Examples
    --------
    @register_nx
    @nxfunc(large_clique_size)
    class NxLargeCliqueSize(NxGraph):pass

    """

    def decorator_func(cls):
        cls.extract = lambda s, data: func(data.G, *s._args, **s._kwargs)
        return cls

    return decorator_func


@register_nx
@nxfunc(large_clique_size)
class NxLargeCliqueSize(NxGraph):
    pass


@register_nx
@nxfunc(average_clustering)
class NxAverageClusteringApproximate(NxGraph):
    pass


@register_nx
@nxfunc(degree_assortativity_coefficient)
class NxDegreeAssortativityCoefficient(NxGraph):
    pass


@register_nx
@nxfunc(degree_pearson_correlation_coefficient)
class NxDegreePearsonCorrelationCoefficient(NxGraph):
    pass


@register_nx
@nxfunc(has_bridges)
class NxHasBridge(NxGraph):
    pass


@register_nx
@nxfunc(graph_clique_number)
class NxGraphCliqueNumber(NxGraph):
    pass


@register_nx
@nxfunc(graph_number_of_cliques)
class NxGraphNumberOfCliques(NxGraph):
    pass


@register_nx
@nxfunc(transitivity)
class NxTransitivity(NxGraph):
    pass


@register_nx
@nxfunc(average_clustering)
class NxAverageClustering(NxGraph):
    pass


@register_nx
@nxfunc(is_connected)
class NxIsConnected(NxGraph):
    pass


@register_nx
@nxfunc(number_connected_components)
class NxNumberConnectedComponents(NxGraph):
    pass


# from networkx.algorithms.components import is_attracting_component
# @register_nx
# @nxfunc(is_attracting_component)
# class NxIsAttractingComponent(NxGraph):pass

# from networkx.algorithms.components import number_attracting_components
# @register_nx
# @nxfunc(number_attracting_components)
# class NxNumberAttractingComponents(NxGraph):pass

# from networkx.algorithms.connectivity.connectivity import average_node_connectivity
# @register_nx
# @nxfunc(average_node_connectivity)
# class NxAverageNodeConnectivity(NxGraph):pass

# from networkx.algorithms.distance_measures import diameter
# @register_nx
# @nxfunc(diameter)
# class NxDiameter(NxGraph):pass

# from networkx.algorithms.distance_measures import radius
# @register_nx
# @nxfunc(radius)
# class NxRadius(NxGraph):pass


@register_nx
@nxfunc(is_distance_regular)
class NxIsDistanceRegular(NxGraph):
    pass


@register_nx
@nxfunc(local_efficiency)
class NxLocalEfficiency(NxGraph):
    pass


@register_nx
@nxfunc(global_efficiency)
class NxGlobalEfficiency(NxGraph):
    pass


@register_nx
@nxfunc(is_eulerian)
class NxIsEulerian(NxGraph):
    pass


# till algorithms.flows
