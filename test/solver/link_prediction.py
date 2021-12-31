from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoLinkPredictor
from autogl.backend import DependentBackend
import numpy as np
import scipy.sparse as sp

if DependentBackend.is_pyg():
    from torch_geometric.utils import train_test_split_edges
    from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset as convert_dataset
    def split_edges(dataset, train, val):
        for i in range(len(dataset)):
            dataset[i] = train_test_split_edges(dataset[i], val, 1 - train - val)
        return dataset
else:
    import dgl
    from autogl.datasets.utils.conversion._to_dgl_dataset import to_dgl_dataset as convert_dataset
    def split_train_test(g, train, val):
        u, v = g.edges()

        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * (1 - train - val))
        val_size = int(len(eids) * val)
        train_size = g.number_of_edges() - test_size - val_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        val_pos_u, val_pos_v = u[eids[test_size : test_size + val_size]], v[eids[test_size : test_size + val_size]]
        train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size: test_size + val_size]], neg_v[neg_eids[test_size: test_size + val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

        train_g = dgl.add_self_loop(dgl.remove_edges(g, eids[:test_size + val_size]))
        # import pdb
        # pdb.set_trace()

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

        val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
        val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

        return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g

    def split_edges(dataset, train, val):
        for i in range(len(dataset)):
            dataset[i] = split_train_test(dataset[i], train, val)
        return dataset

from autogl.datasets.utils import split_edges

cora = build_dataset_from_name("cora")
cora = convert_dataset(cora)
cora = split_edges(cora, 0.8, 0.05)

solver = AutoLinkPredictor(
    graph_models=("gin", "gat", "gcn"),
    hpo_module=None,
    device="auto"
)

solver.fit(cora, evaluation_method=["acc"])
solver.leaderboard.show()
result = solver.predict(cora)

print(result)
