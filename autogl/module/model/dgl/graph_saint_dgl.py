import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn
import math
import os
import time
import torch as th
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, order=1, act=None,
                 dropout=0, batch_norm=False, aggr="concat"):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        self.bias = nn.ParameterList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bias.append(nn.Parameter(th.zeros(out_dim)))

        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.batch_norm = batch_norm
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(order + 1):
                self.offset.append(nn.Parameter(th.zeros(out_dim)))
                self.scale.append(nn.Parameter(th.ones(out_dim)))

        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)

    def feat_trans(self, features, idx):
        h = self.lins[idx](features) + self.bias[idx]

        if self.act is not None:
            h = self.act(h)

        if self.batch_norm:
            mean = h.mean(dim=1).view(h.shape[0], 1)
            var = h.var(dim=1, unbiased=False).view(h.shape[0], 1) + 1e-9
            h = (h - mean) * self.scale[idx] * th.rsqrt(var) + self.offset[idx]

        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        D_norm = g.ndata['train_D_norm'] if 'train_D_norm' in g.ndata else g.ndata['full_D_norm']
        for _ in range(self.order):
            g.ndata['h'] = h_hop[-1]
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * D_norm
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = th.cat(h_part, 1)
        else:
            raise NotImplementedError

        return h_out


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, arch="1-1-0",
                 act=F.relu, dropout=0, batch_norm=False, aggr="concat"):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split('-')))
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, order=orders[0],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[0] + 1) * hid_dim

        for i in range(1, len(orders)-1):
            self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[i],
                                     act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
            pre_out = ((aggr == "concat") * orders[i] + 1) * hid_dim

        self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[-1],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[-1] + 1) * hid_dim

        self.out_layer = GCNLayer(in_dim=pre_out, out_dim=out_dim, order=0,
                                  act=None, dropout=dropout, batch_norm=False, aggr=aggr)

    def forward(self, graph):
        h = graph.ndata['feat']

        for layer in self.gcn:
            h = layer(graph, h)

        h = F.normalize(h, p=2, dim=1)
        h = self.out_layer(graph, h)

        return h




# The base class of sampler
# (TODO): online sampling
class SAINTSampler(object):
    def __init__(self, dn, g, train_nid, node_budget, num_repeat=50):
        """
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes
        :param num_repeat: number of times of repeating sampling one node
        """
        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn, self.num_repeat = dn, num_repeat
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None

        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            aggr_norm, loss_norm = np.load(norm_fn, allow_pickle=True)
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0

            t = time.perf_counter()
            while sampled_nodes <= self.train_g.num_nodes() * num_repeat:
                subgraph = self.__sample__()
                self.subgraphs.append(subgraph)
                sampled_nodes += subgraph.shape[0]
                self.N += 1
            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs)

            t = time.perf_counter()
            self.__counter__()
            aggr_norm, loss_norm = self.__compute_norm__()
            print(f'Normalization time: [{time.perf_counter() - t:.2f}s]')
            np.save(norm_fn, (aggr_norm, loss_norm))

        self.train_g.ndata['l_n'] = th.Tensor(loss_norm)
        self.train_g.edata['w'] = th.Tensor(aggr_norm)
        self.__compute_degree_norm()

        self.num_batch = math.ceil(self.train_g.num_nodes() / node_budget)
        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))
        print("The size of subgraphs is about: ", len(self.subgraphs[-1]))

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __counter__(self):

        for sampled_nodes in self.subgraphs:
            sampled_nodes = th.from_numpy(sampled_nodes)
            self.node_counter[sampled_nodes] += 1

            subg = self.train_g.subgraph(sampled_nodes)
            sampled_edges = subg.edata[dgl.EID]
            self.edge_counter[sampled_edges] += 1

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):
        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_batch:
            result = self.train_g.subgraph(self.subgraphs[self.n])
            self.n += 1
            return result
        else:
            random.shuffle(self.subgraphs)
            raise StopIteration()


class SAINTNodeSampler(SAINTSampler):
    def __init__(self, node_budget, dn, g, train_nid, num_repeat=50):
        self.node_budget = node_budget
        super(SAINTNodeSampler, self).__init__(dn, g, train_nid, node_budget, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Node_{}_{}.npy'.format(self.dn, self.node_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_Node_{}_{}_norm.npy'.format(self.dn, self.node_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            self.prob = self.train_g.in_degrees().float().clamp(min=1)

        sampled_nodes = th.multinomial(self.prob, num_samples=self.node_budget, replacement=True).unique()
        return sampled_nodes.numpy()


class SAINTEdgeSampler(SAINTSampler):
    def __init__(self, edge_budget, dn, g, train_nid, num_repeat=50):
        self.edge_budget = edge_budget
        super(SAINTEdgeSampler, self).__init__(dn, g, train_nid, edge_budget * 2, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_Edge_{}_{}.npy'.format(self.dn, self.edge_budget,
                                                                       self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_Edge_{}_{}_norm.npy'.format(self.dn, self.edge_budget,
                                                                           self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        if self.prob is None:
            src, dst = self.train_g.edges()
            src_degrees, dst_degrees = self.train_g.in_degrees(src).float().clamp(min=1),\
                                       self.train_g.in_degrees(dst).float().clamp(min=1)
            self.prob = 1. / src_degrees + 1. / dst_degrees

        sampled_edges = th.multinomial(self.prob, num_samples=self.edge_budget, replacement=True).unique()

        sampled_src, sampled_dst = self.train_g.find_edges(sampled_edges)
        sampled_nodes = th.cat([sampled_src, sampled_dst]).unique()
        return sampled_nodes.numpy()


class SAINTRandomWalkSampler(SAINTSampler):
    def __init__(self, num_roots, length, dn, g, train_nid, num_repeat=50):
        self.num_roots, self.length = num_roots, length
        super(SAINTRandomWalkSampler, self).__init__(dn, g, train_nid, num_roots * length, num_repeat)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_repeat))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_repeat))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots, ))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()
