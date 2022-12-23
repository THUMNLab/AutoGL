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
from .._feature_engineer import FeatureEngineer
from ..._data_preprocessor_registry import DataPreprocessorUniversalRegistry


class _NetworkXGraphFeatureEngineer(FeatureEngineer):
    def __init__(self, feature_extractor: _typing.Callable[[networkx.Graph], _typing.Any]):
        self.__feature_extractor: _typing.Callable[[networkx.Graph], _typing.Any] = feature_extractor
        super(_NetworkXGraphFeatureEngineer, self).__init__()

    def __transform_homogeneous_static_graph(
            self, homogeneous_static_graph: GeneralStaticGraph
    ) -> GeneralStaticGraph:
        if not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError("Provided static graph must be homogeneous")
        dsc: torch.Tensor = torch.tensor(
            [
                self.__feature_extractor(
                    conversion.HomogeneousStaticGraphToNetworkX(to_undirected=True)(homogeneous_static_graph)
                )
            ]
        ).view(-1)
        if 'gf' in homogeneous_static_graph.data:
            gf = homogeneous_static_graph.data['gf'].view(-1)
            homogeneous_static_graph.data['gf'] = torch.cat([gf, dsc])
        else:
            homogeneous_static_graph.data['gf'] = dsc
        return homogeneous_static_graph

    @classmethod
    def __edge_index_to_nx_graph(cls, edge_index: torch.Tensor) -> networkx.Graph:
        g: networkx.Graph = networkx.Graph()
        for u, v in edge_index.t().tolist():
            if u == v:
                continue
            else:
                g.add_edge(u, v)
        return g

    def __transform_data(self, data):
        if not (
                hasattr(data, "edge_index") and
                torch.is_tensor(data.edge_index) and
                isinstance(data.edge_index, torch.Tensor) and
                data.edge_index.dim() == data.edge_index.size(0) == 2 and
                data.edge_index.dtype == torch.long
        ):
            raise TypeError("Unsupported provided data")
        dsc: torch.Tensor = torch.tensor(
            [self.__feature_extractor(self.__edge_index_to_nx_graph(data.edge_index))]
        ).view(-1)
        if hasattr(data, 'gf') and isinstance(data.gf, torch.Tensor):
            gf = data.gf.view(-1)
            data.gf = torch.cat([gf, dsc])
        else:
            data.gf = dsc
        return data

    def _transform(
            self, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        if isinstance(data, GeneralStaticGraph):
            return self.__transform_homogeneous_static_graph(data)
        else:
            return self.__transform_data(data)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXLargeCliqueSize")
class NXLargeCliqueSize(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXLargeCliqueSize, self).__init__(large_clique_size)


# @FeatureEngineerUniversalRegistry.register_feature_engineer("NXAverageClusteringApproximate")
# class NXAverageClusteringApproximate(_NetworkXGraphFeatureEngineer):
#     def __init__(self):
#         super(NXAverageClusteringApproximate, self).__init__(average_clustering)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXDegreeAssortativityCoefficient")
class NXDegreeAssortativityCoefficient(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXDegreeAssortativityCoefficient, self).__init__(degree_assortativity_coefficient)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXDegreePearsonCorrelationCoefficient")
class NXDegreePearsonCorrelationCoefficient(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXDegreePearsonCorrelationCoefficient, self).__init__(degree_pearson_correlation_coefficient)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXHasBridges")
class NXHasBridges(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXHasBridges, self).__init__(has_bridges)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXGraphCliqueNumber")
class NXGraphCliqueNumber(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGraphCliqueNumber, self).__init__(graph_clique_number)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXGraphNumberOfCliques")
class NXGraphNumberOfCliques(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGraphNumberOfCliques, self).__init__(graph_number_of_cliques)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXTransitivity")
class NXTransitivity(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXTransitivity, self).__init__(transitivity)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXAverageClustering")
class NXAverageClustering(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXAverageClustering, self).__init__(average_clustering)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXIsConnected")
class NXIsConnected(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsConnected, self).__init__(is_connected)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXNumberConnectedComponents")
class NXNumberConnectedComponents(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXNumberConnectedComponents, self).__init__(number_connected_components)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXIsDistanceRegular")
class NXIsDistanceRegular(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsDistanceRegular, self).__init__(is_distance_regular)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXLocalEfficiency")
class NXLocalEfficiency(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXLocalEfficiency, self).__init__(local_efficiency)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXGlobalEfficiency")
class NXGlobalEfficiency(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXGlobalEfficiency, self).__init__(global_efficiency)


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NXIsEulerian")
class NXIsEulerian(_NetworkXGraphFeatureEngineer):
    def __init__(self):
        super(NXIsEulerian, self).__init__(is_eulerian)
