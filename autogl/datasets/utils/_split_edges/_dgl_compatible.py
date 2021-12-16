import numpy as np
import scipy.sparse as sp
import torch
import typing as _typing
import dgl
import autogl.data.graph
from .train_test_split_edges import train_test_split_edges
from autogl.data.graph.utils.conversion import static_graph_to_general_data


class _SplitEdgesDGLImpl:
    @classmethod
    def __split_edges_train_val_test(
            cls, g: dgl.DGLGraph,
            train_ratio: float, val_ratio: float
    ) -> _typing.Tuple[
        dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph
    ]:
        u, v = g.edges()

        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)

        valid_size = int(len(eids) * val_ratio)
        test_size = int(len(eids) * (1 - train_ratio - val_ratio))
        train_size = g.number_of_edges() - test_size - valid_size

        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        valid_pos_u, valid_pos_v = u[eids[test_size:test_size + valid_size]], v[eids[test_size:test_size + valid_size]]
        train_pos_u, train_pos_v = u[eids[test_size + valid_size:]], v[eids[test_size + valid_size:]]

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        valid_neg_u, valid_neg_v = neg_u[neg_eids[test_size:test_size + valid_size]], neg_v[neg_eids[test_size:test_size + valid_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[train_size:]], neg_v[neg_eids[train_size:]]

        train_g = dgl.remove_edges(g, eids[:test_size + valid_size])

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

        valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=g.number_of_nodes())
        valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

        return (
            train_g, train_pos_g, train_neg_g,
            valid_pos_g, valid_neg_g, test_pos_g, test_neg_g
        )

    @classmethod
    def __split_edges_train_test(
            cls, g: dgl.DGLGraph, train_ratio: float
    ) -> _typing.Tuple[
        dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph, dgl.DGLGraph,
    ]:
        u, v = g.edges()

        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * (1 - train_ratio))
        train_size = g.number_of_edges() - test_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[train_size:]], neg_v[neg_eids[train_size:]]

        train_g = dgl.remove_edges(g, eids[:test_size])

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

        return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

    @classmethod
    def split_edges_for_dgl_graph(
            cls, graph: dgl.DGLGraph,
            train_ratio: float, val_ratio: _typing.Optional[float] = ...
    ) -> _typing.Union[
        _typing.Tuple[
            dgl.DGLGraph, dgl.DGLGraph,
            dgl.DGLGraph, dgl.DGLGraph,
            dgl.DGLGraph, dgl.DGLGraph,
            dgl.DGLGraph
        ],
        _typing.Tuple[
            dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph,
            dgl.DGLGraph, dgl.DGLGraph,
        ]
    ]:
        if not 0 < train_ratio < 1:
            raise ValueError(f"Invalid train_ratio as {train_ratio}")
        if isinstance(val_ratio, float):
            if not 0 < val_ratio < 1:
                raise ValueError(f"Invalid val_ratio as {val_ratio}")
            if not 0 < train_ratio + val_ratio < 1:
                raise ValueError(
                    f"Invalid combination (train_ratio, val_ratio) "
                    f"as ({train_ratio}, {val_ratio})"
                )
            return cls.__split_edges_train_val_test(graph, train_ratio, val_ratio)
        else:
            return cls.__split_edges_train_test(graph, train_ratio)


def split_edges_for_data(
        data: _typing.Union[
            dgl.DGLGraph, autogl.data.graph.GeneralStaticGraph, _typing.Any
        ],
        train_ratio: float, val_ratio: _typing.Optional[float]
) -> _typing.Union[
    autogl.data.Data,
    _typing.Tuple[
        dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph
    ],
    _typing.Tuple[
        dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph,
        dgl.DGLGraph, dgl.DGLGraph,
    ]
]:
    if isinstance(data, dgl.DGLGraph):
        if not data.is_homogeneous:
            raise ValueError(
                "provided DGL graph to split edges MUST be homogeneous"
            )
        else:
            return _SplitEdgesDGLImpl.split_edges_for_dgl_graph(
                data, train_ratio, val_ratio
            )
    elif isinstance(data, autogl.data.graph.GeneralStaticGraph):
        if not (data.nodes.is_homogeneous and data.edges.is_homogeneous):
            raise ValueError(
                "Provided instance of GeneralStaticGraph MUST be homogeneous"
            )
        __data = static_graph_to_general_data(data)
        edge_index: torch.LongTensor = data.edges.connections
        edge_attr: _typing.Optional[torch.Tensor] = (
            data.edges.data['edge_attr'] if 'edge_attr' in data.edges.data else None
        )
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
            y: _typing.Optional[torch.Tensor] = data.x
        else:
            y: _typing.Optional[torch.Tensor] = None
        __data = autogl.data.Data(
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
