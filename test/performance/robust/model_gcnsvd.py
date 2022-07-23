import os
import pickle
from torchaudio import datasets
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCNSVD
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg, AmazonPyg
import argparse

os.environ["AUTOGL_BACKEND"] = "pyg"


from autogl.module.model.pyg import AutoGCNSVD
from autogl.solver.utils import set_seed

def test(model, data, mask):
    model.eval()

    if hasattr(model, 'cls_forward'):
        out = model.cls_forward(data)[mask]
    else:
        out = model(data)[mask]
    pred = out.max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc

def train(model, data, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(data)
    parameters = model.state_dict()
    best_acc = 0.
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        if hasattr(model, 'cls_forward'):
            output = model.cls_forward(data)
        else:
            output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_acc = test(model, data, data.val_mask)
        if val_acc > best_acc:
            best_acc = val_acc
            parameters = pickle.dumps(model.state_dict())
            
    model.load_state_dict(pickle.loads(parameters))
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser('pyg model')
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
    parser.add_argument('--k', type=int, default=15, help='Truncated Components.')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('cuda: %s' % args.cuda)

    # make sure you use the same data splits as you generated attacks
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Here the random seed is to split the train/val/test data,
    # we need to set the random seed to be the same as that when you generate the perturbed graph
    # data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15)
    # Or we can just use setting='prognn' to get the splits
    data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    pyg_data = Dpr2Pyg(data).process().to(args.device)
    pyg_data.num_classes = len(set(labels))

    # load pre-attacked graph
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset,
            attack_method='meta',
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj

    print('=== testing GCN-SVD on perturbed graph (AutoGL) ===')
    model_hp = {
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.5,
            "act": "relu"
        }
    accs = []
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        model = AutoGCNSVD(
                num_features=pyg_data.num_node_features,
                num_classes=pyg_data.num_classes,
                device=args.device,
                init=False
            ).from_hyper_parameter(model_hp).model
        model.to(args.device)

        train(model, pyg_data, args)
        acc = test(model, pyg_data, pyg_data.test_mask)
        accs.append(acc)
    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))


    print('=== testing GCN-SVD on perturbed graph (deeprobust)===')
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1,
                    nhid=16, device=args.device)

    model = model.to(args.device)
    # Test set results: loss= 0.8541 accuracy= 0.7067
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=args.k, verbose=True)
    model.eval()
    output = model.test(idx_test)
    print(output)
    