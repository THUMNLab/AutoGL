import os
import dgl.data.utils
import numpy as np
import scipy.io
import torch
from autogl.data import InMemoryStaticGraphSet
from .. import _dataset_registry


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()


@_dataset_registry.DatasetUniversalRegistry.register_dataset("hetero-acm-han")
class ACMHANDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        data_path: str = os.path.join(path, 'raw', 'ACM.mat')
        _url: str = "https://data.dgl.ai/dataset/ACM.mat"
        if os.path.exists(data_path) and os.path.isfile(data_path):
            print(f"Using cached file {data_path}")
        else:
            dgl.data.utils.download(_url, data_path)
        data = scipy.io.loadmat(data_path)
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        hg = dgl.heterograph({
            ('paper', 'pa', 'author'): p_vs_a.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_l.nonzero(),
            ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
        })
        hg.nodes['paper'].data['feat'] = torch.tensor(p_vs_t.toarray(), dtype=torch.float)

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        hg.nodes['paper'].data['label'] = torch.LongTensor(labels)

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_nodes = hg.number_of_nodes('paper')
        hg.nodes['paper'].data['train_mask'] = get_binary_mask(num_nodes, train_idx)
        hg.nodes['paper'].data['val_mask'] = get_binary_mask(num_nodes, val_idx)
        hg.nodes['paper'].data['test_mask'] = get_binary_mask(num_nodes, test_idx)

        super(ACMHANDataset, self).__init__([hg])
        self.schema.meta_paths = (('pa', 'ap'), ('pf', 'fp'))
        self.schema['target_node_type'] = 'paper'


@_dataset_registry.DatasetUniversalRegistry.register_dataset("hetero-acm-hgt")
class ACMHGTDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        data_path: str = os.path.join(path, 'raw', 'ACM.mat')
        _url: str = "https://data.dgl.ai/dataset/ACM.mat"
        if os.path.exists(data_path) and os.path.isfile(data_path):
            print(f"Using cached file {data_path}")
        else:
            dgl.data.utils.download(_url, data_path)
        data = scipy.io.loadmat(data_path)

        hg = dgl.heterograph({
            ('paper', 'written-by', 'author'): data['PvsA'].nonzero(),
            ('author', 'writing', 'paper'): data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper'): data['PvsP'].nonzero(),
            ('paper', 'cited', 'paper'): data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject'): data['PvsL'].nonzero(),
            ('subject', 'has', 'paper'): data['PvsL'].transpose().nonzero(),
        })

        pvc = data['PvsC'].tocsr()
        p_selected = pvc.tocoo()
        # generate labels
        labels = pvc.indices
        hg.nodes['paper'].data['label'] = torch.tensor(labels).long()

        # generate train/val/test split
        pid = p_selected.row
        shuffle = np.random.permutation(pid)
        train_idx = torch.tensor(shuffle[0:800]).long()
        val_idx = torch.tensor(shuffle[800:900]).long()
        test_idx = torch.tensor(shuffle[900:]).long()
        num_nodes = hg.number_of_nodes('paper')
        hg.nodes['paper'].data['train_mask'] = get_binary_mask(num_nodes, train_idx)
        hg.nodes['paper'].data['val_mask'] = get_binary_mask(num_nodes, val_idx)
        hg.nodes['paper'].data['test_mask'] = get_binary_mask(num_nodes, test_idx)

        hg.node_dict = {}
        hg.edge_dict = {}
        for node_type in hg.ntypes:
            hg.node_dict[node_type] = len(hg.node_dict)
        for edge_type in hg.etypes:
            hg.edge_dict[edge_type] = len(hg.edge_dict)

        for edge_type in hg.etypes:
            hg.edges[edge_type].data['id'] = (
                    torch.ones(hg.number_of_edges(edge_type), dtype=torch.long) * len(hg.edge_dict)
            )

        # Random initialize input feature
        for node_type in hg.ntypes:
            embeddings = torch.Tensor(hg.number_of_nodes(node_type), 256)
            torch.nn.init.xavier_uniform_(embeddings)
            hg.nodes[node_type].data['feat'] = embeddings

        super(ACMHGTDataset, self).__init__([hg])
        self.schema['target_node_type'] = 'paper'
