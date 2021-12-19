import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import argparse
from tqdm import tqdm


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



def test(model, graph, idx, labels):
    model.eval()
    pred = model(graph,'paper')[idx].max(1)[1].cpu()
    acc = (pred == labels[idx]).float().mean()
    return acc

def train(model, G):
    best_val_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'paper')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(args.device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        # if epoch % 5 == 0:
        #     model.eval()
        #     logits = model(G, 'paper')
        #     pred   = logits.argmax(1).cpu()
        #     train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        #     val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
        #     test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
        #     if best_val_acc < val_acc:
        #         best_val_acc = val_acc
        #         best_test_acc = test_acc
            # print('Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            #     epoch,
            #     optimizer.param_groups[0]['lr'], 
            #     loss.item(),
            #     train_acc.item(),
            #     val_acc.item(),
            #     best_val_acc.item(),
            #     test_acc.item(),
            #     best_test_acc.item(),
            # ))





if __name__=='__main__':
    torch.manual_seed(0)
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/tmp/ACM.mat'

    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)

    parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--n_hid',   type=int, default=256)
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--model', type=str, choices=['hgt', 'HeteroRGCN'], default='hgt')

    args = parser.parse_args()
    # device = torch.device("cuda:0")

    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        })
    print(G)

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # generate labels
    labels = pvc.indices
    labels = torch.tensor(labels).long()

    # generate train/val/test split
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()

    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

    #     Random initialize input feature
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['inp'] = emb

    G = G.to(args.device)
    accs = []
    for seed in tqdm(range(args.repeat)):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if args.model == 'HeteroRGCN':
            model = HeteroRGCN(G,
                   in_size=args.n_inp,
                   hidden_size=args.n_hid,
                   out_size=labels.max().item()+1)
        elif args.model == 'hgt':
            model = HGT(G,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=2,
                n_heads=4)
        model.to(args.device)

        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr = args.max_lr)
        # print('Training with #param: %d' % (get_n_params(model)))

        train(model, G)
        acc = test(model, G, test_idx, labels)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))