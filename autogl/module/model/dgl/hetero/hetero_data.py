import dgl
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from scipy import io as sio 
import os.path as osp
import sys 
import torch 
import numpy as np
import urllib.request
import torch.nn as nn

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

class BaseHeteroDataset():
    r"""
    Description
    -----------
    
    
    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The dgl heterogeneous graph.
    num_classes : int
        Number of classes for target nodes.
    metapaths : List[[List[str]]]
    """
    def __init__(self,):
        self.num_classes = None 
        self.metapaths = None
        self.num_features = None
        self.g = None


class HeteroData(BaseHeteroDataset):
    
    def __init__(self, name, **kwargs):
        super(HeteroData, self).__init__()
        self.name = name 

        if name=='acm_raw':
            self.g, self.num_classes, self.num_features = self.load_acm_raw()
        elif name=='acm':
            self.g, self.num_classes, self.num_features = self.load_hgt_acm(random_init_fea=True)

    def load_acm_raw(self):
        self.metapaths = [['pa', 'ap'], ['pf', 'fp']]
        filename = 'ACM.mat'
        url = 'dataset/' + filename
        data_path = get_download_dir() + '/' + filename
        if osp.exists(data_path):
            print(f'Using existing file {filename}', file=sys.stderr)
        else:
            download(_get_dgl_url(url), path=data_path)

        data = sio.loadmat(data_path)
        p_vs_l = data['PvsL']       # paper-field?
        p_vs_a = data['PvsA']       # paper-author
        p_vs_t = data['PvsT']       # paper-term, bag of words
        p_vs_c = data['PvsC']       # paper-conference, labels come from that

        # We assign
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

        hg.nodes['paper'].data['feat'] = torch.FloatTensor(p_vs_t.toarray())
        #features = torch.FloatTensor(p_vs_t.toarray())

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        hg.nodes['paper'].data['label'] = torch.LongTensor(labels)

        num_classes = 3

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

        num_features = hg.nodes['paper'].data['feat'].size(1)

        return hg, num_classes, num_features

    def load_hgt_acm(self, random_init_fea=True):
        data_url = 'https://data.dgl.ai/dataset/ACM.mat'
        data_file_path = '/tmp/ACM.mat'

        urllib.request.urlretrieve(data_url, data_file_path)
        data = sio.loadmat(data_file_path)

        hg = dgl.heterograph({
                ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
                ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
                ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
                ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
                ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
                ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
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
        for ntype in hg.ntypes:
           hg.node_dict[ntype] = len(hg.node_dict)
        for etype in hg.etypes:
            hg.edge_dict[etype] = len(hg.edge_dict)

        for etype in hg.etypes:
            hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * len(hg.edge_dict)

        # Random initialize input feature
        if random_init_fea:
            for ntype in hg.ntypes:
                emb = nn.Parameter(torch.Tensor(hg.number_of_nodes(ntype), 256), requires_grad = False)
                nn.init.xavier_uniform_(emb)
                hg.nodes[ntype].data['feat'] = emb

        num_features = 256
        num_classes = labels.max().item()+1

        return hg, num_classes, num_features

    

    
if __name__=='__main__':
    HeteroData('acm_raw')
    HeteroData('acm')