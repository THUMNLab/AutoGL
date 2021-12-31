import torch
import typing as _typing
import torch_geometric
import autogl
from autogl.data.graph import GeneralStaticGraph
from autogl.data.graph.utils.conversion import static_graph_to_pyg_data
from .train_test_split_edges import train_test_split_edges


def split_edges_for_data(
        data: _typing.Union[
            torch_geometric.data.Data, autogl.data.graph.GeneralStaticGraph, _typing.Any
        ],
        train_ratio: float, val_ratio: float
) -> torch_geometric.data.Data:
    if isinstance(data, torch_geometric.data.Data):
        if not (
                isinstance(data.edge_index, torch.Tensor) and
                data.edge_index.dim() == data.edge_index.size(0) == 2
        ):
            raise ValueError
        edge_index: torch.LongTensor = data.edge_index
        edge_attr: _typing.Optional[torch.Tensor] = data.edge_attr
        import copy
        __data = copy.copy(data)
    elif isinstance(data, autogl.data.graph.GeneralStaticGraph):
        if not (data.nodes.is_homogeneous and data.edges.is_homogeneous):
            raise ValueError(
                "Provided instance of GeneralStaticGraph MUST be homogeneous"
            )
        edge_index: torch.LongTensor = data.edges.connections
        edge_attr: _typing.Optional[torch.Tensor] = (
            data.edges.data['edge_attr'] if 'edge_attr' in data.edges.data else None
        )
        __data = static_graph_to_pyg_data(data)
    elif (
            hasattr(data, 'edge_index') and
            isinstance(data.edge_index, torch.Tensor) and
            data.edge_index.dim() == data.edge_index.size(0) == 2
    ):
        edge_index: torch.LongTensor = data.edge_index
        if (
                hasattr(data, 'edge_attr') and
                isinstance(data.edge_attr, torch.Tensor) and
                data.edge_attr.size(0) == edge_index.size(1)
        ):
            edge_attr: _typing.Optional[torch.Tensor] = data.edge_attr
        else:
            edge_attr: _typing.Optional[torch.Tensor] = None
        if hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            x: _typing.Optional[torch.Tensor] = data.x
        else:
            x: _typing.Optional[torch.Tensor] = None
        if hasattr(data, 'y') and isinstance(data.y, torch.Tensor):
            y: _typing.Optional[torch.Tensor] = data.y
        else:
            y: _typing.Optional[torch.Tensor] = None
        __data = torch_geometric.data.Data(
            edge_index=edge_index, edge_attr=edge_attr, x=x, y=y
        )
    else:
        raise ValueError

    if isinstance(val_ratio, float) and 0 < val_ratio < 1:
        test_ratio = 1 - train_ratio - val_ratio
    else:
        test_ratio = 1 - train_ratio
    compound_results = train_test_split_edges(
        edge_index, edge_attr,
        val_ratio=val_ratio, test_ratio=test_ratio
    )
    __data.train_pos_edge_index = compound_results.train_pos_edge_index
    __data.train_pos_edge_attr = compound_results.train_pos_edge_attr
    __data.train_neg_adj_mask = compound_results.train_neg_adj_mask
    __data.val_pos_edge_index = compound_results.val_pos_edge_index
    __data.val_pos_edge_attr = compound_results.val_pos_edge_attr
    __data.val_neg_edge_index = compound_results.val_neg_edge_index
    __data.test_pos_edge_index = compound_results.test_pos_edge_index
    __data.test_pos_edge_attr = compound_results.test_pos_edge_attr
    __data.test_neg_edge_index = compound_results.test_neg_edge_index
    return __data
