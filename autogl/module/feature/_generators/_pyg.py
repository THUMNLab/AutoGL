import torch.nn.functional
import autogl
from ._basic import BaseFeatureGenerator
from ._pyg_impl import degree, scatter_min, scatter_max, scatter_mean, scatter_std
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


@FeatureEngineerUniversalRegistry.register_feature_engineer("LocalDegreeProfile")
class LocalDegreeProfileGenerator(BaseFeatureGenerator):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        row, col = data.edge_index
        if data.x is not None and isinstance(data.x, torch.Tensor):
            N = data.x.size(0)
        else:
            N = (torch.max(data.edge_index).item() + 1)

        deg = degree(row, N, dtype=torch.float)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)
        return x


@FeatureEngineerUniversalRegistry.register_feature_engineer("NormalizeFeatures")
class NormalizeFeatures(BaseFeatureGenerator):
    def __init__(self):
        super(NormalizeFeatures, self).__init__(override_features=True)

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        if data.x is not None and isinstance(data.x, torch.Tensor):
            data.x.div_(data.x.sum(dim=-1, keepdim=True).clamp_(min=1.))
        return data.x


@FeatureEngineerUniversalRegistry.register_feature_engineer("OneHotDegree")
class OneHotDegreeGenerator(BaseFeatureGenerator):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(
            self, max_degree: int = 1000,
            in_degree: bool = False, cat: bool = True
    ):
        self.__max_degree: int = max_degree
        self.__in_degree: bool = in_degree
        self.__cat: bool = cat
        super(OneHotDegreeGenerator, self).__init__()

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        idx, x = data.edge_index[1 if self.__in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = torch.nn.functional.one_hot(
            deg, num_classes=self.__max_degree + 1
        ).to(torch.float)
        return deg
