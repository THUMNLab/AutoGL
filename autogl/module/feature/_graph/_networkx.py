import torch
import typing as _typing
import networkx
from networkx.algorithms.euler import is_eulerian
from networkx.algorithms.efficiency_measures import global_efficiency
from networkx.algorithms.efficiency_measures import local_efficiency
from networkx.algorithms.distance_regular import is_distance_regular
from networkx.algorithms.components import number_connected_components
from networkx.algorithms.components import is_connected
# from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.clique import graph_number_of_cliques
from networkx.algorithms.clique import graph_clique_number
from networkx.algorithms.bridges import has_bridges
from networkx.algorithms.assortativity import degree_pearson_correlation_coefficient
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.approximation.clique import large_clique_size

from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion
from .._base_feature_engineer import BaseFeatureEngineer
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


class _NetworkXGraphFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, feature_extractor: _typing.Callable[[networkx.Graph], _typing.Any]):
        self.__feature_extractor: _typing.Callable[[networkx.Graph], _typing.Any] = feature_extractor
        super(_NetworkXGraphFeatureEngineer, self).__init__()

    def _transform(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        dsc = self.__feature_extractor(
            conversion.HomogeneousStaticGraphToNetworkX(to_undirected=True)(static_graph)
        )
        dsc: torch.Tensor = torch.tensor([dsc]).view(-1)
        if 'gf' in static_graph.data:
            gf = static_graph.data['gf'].view(-1)
            static_graph.data['gf'] = torch.cat([gf, dsc])
        else:
            static_graph.data['gf'] = dsc
        return static_graph


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXLargeCliqueSize")
class NXLargeCliqueSize(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXLargeCliqueSize, self).__init__(large_clique_size)


# @FeatureEngineerUniversalRegistry.register_feature_engineer("NXAverageClusteringApproximate")
# class NXAverageClusteringApproximate(_NetworkXGraphFeatureEngineer):
#     def __init__(self):
#         super(NXAverageClusteringApproximate, self).__init__(average_clustering)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXDegreeAssortativityCoefficient")
class NXDegreeAssortativityCoefficient(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXDegreeAssortativityCoefficient, self).__init__(degree_assortativity_coefficient)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXDegreePearsonCorrelationCoefficient")
class NXDegreePearsonCorrelationCoefficient(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXDegreePearsonCorrelationCoefficient, self).__init__(degree_pearson_correlation_coefficient)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXHasBridges")
class NXHasBridges(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXHasBridges, self).__init__(has_bridges)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXGraphCliqueNumber")
class NXGraphCliqueNumber(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGraphCliqueNumber, self).__init__(graph_clique_number)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXGraphNumberOfCliques")
class NXGraphNumberOfCliques(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGraphNumberOfCliques, self).__init__(graph_number_of_cliques)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXTransitivity")
class NXTransitivity(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXTransitivity, self).__init__(transitivity)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXAverageClustering")
class NXAverageClustering(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXAverageClustering, self).__init__(average_clustering)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXIsConnected")
class NXIsConnected(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsConnected, self).__init__(is_connected)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXNumberConnectedComponents")
class NXNumberConnectedComponents(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXNumberConnectedComponents, self).__init__(number_connected_components)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXIsDistanceRegular")
class NXIsDistanceRegular(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsDistanceRegular, self).__init__(is_distance_regular)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXLocalEfficiency")
class NXLocalEfficiency(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXLocalEfficiency, self).__init__(local_efficiency)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXGlobalEfficiency")
class NXGlobalEfficiency(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGlobalEfficiency, self).__init__(global_efficiency)


@FeatureEngineerUniversalRegistry.register_feature_engineer("NXIsEulerian")
class NXIsEulerian(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsEulerian, self).__init__(is_eulerian)
