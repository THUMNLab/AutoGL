import netlsd
import torch

from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils import conversion
from .._base_feature_engineer import BaseFeatureEngineer
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


@FeatureEngineerUniversalRegistry.register_feature_engineer("NetLSD".lower())
class NetLSD(BaseFeatureEngineer):
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

    def _transform(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        temp = netlsd.heat(
            conversion.HomogeneousStaticGraphToNetworkX(to_undirected=True).__call__(
                static_graph, to_undirected=True
            ),
            *self.__args, **self.__kwargs
        )
        dsc: torch.Tensor = torch.tensor([temp]).view(-1)
        if 'gf' in static_graph.data:
            gf = static_graph.data['gf'].view(-1)
            static_graph.data['gf'] = torch.cat([gf, dsc])
        else:
            static_graph.data['gf'] = dsc
        return static_graph
