import netlsd
import networkx
import torch
import typing
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion
from .._feature_engineer import FeatureEngineer
from ..._data_preprocessor_registry import DataPreprocessorUniversalRegistry


@DataPreprocessorUniversalRegistry.register_data_preprocessor("NetLSD".lower())
class NetLSD(FeatureEngineer):
    r"""
    Notes
    -----
    a graph feature generation method. This is a simple wrapper of NetLSD [#]_.

    References
    ----------
    ..  [#] A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, and E. Müller, “NetLSD: Hearing the shape of a graph,”
        Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., pp. 2347–2356, 2018.
    """

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        super(NetLSD, self).__init__()

    def __extract(self, nx_g: networkx.Graph) -> torch.Tensor:
        return torch.tensor(netlsd.heat(nx_g, *self.__args, **self.__kwargs)).view(-1)

    def __transform_homogeneous_static_graph(
            self, homogeneous_static_graph: GeneralStaticGraph
    ) -> GeneralStaticGraph:
        if not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError("Provided static graph must be homogeneous")
        dsc: torch.Tensor = self.__extract(
            conversion.HomogeneousStaticGraphToNetworkX(to_undirected=True).__call__(
                homogeneous_static_graph, to_undirected=True
            )
        )
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
        dsc: torch.Tensor = self.__extract(self.__edge_index_to_nx_graph(data.edge_index))
        if hasattr(data, 'gf') and isinstance(data.gf, torch.Tensor):
            gf = data.gf.view(-1)
            data.gf = torch.cat([gf, dsc])
        else:
            data.gf = dsc
        return data

    def _transform(
            self, data: typing.Union[GeneralStaticGraph, typing.Any]
    ) -> typing.Union[GeneralStaticGraph, typing.Any]:
        if isinstance(data, GeneralStaticGraph):
            return self.__transform_homogeneous_static_graph(data)
        else:
            return self.__transform_data(data)
