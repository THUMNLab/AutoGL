"""
Performance check of AutoGL model + PYG (trainer + dataset)
"""
import os
import pickle
import numpy as np
from tqdm import tqdm

os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from autogl.module.model.pyg import AutoGCN, AutoGAT, AutoSAGE
from autogl.datasets import utils
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging

logging.basicConfig(level=logging.ERROR)

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

def init_model(model_name, model_hp, num_node_features, num_classes):
    if model_name == 'gat':
        model = AutoGAT(
            num_features=num_node_features,
            num_classes=num_classes,
            device=args.device,
            init=False
        ).from_hyper_parameter(model_hp).model
    elif model_name == 'gcn':
        model = AutoGCN(
            num_features=num_node_features,
            num_classes=num_classes,
            device=args.device,
            init=False
        ).from_hyper_parameter(model_hp).model
    elif model_name == 'sage':
        model = AutoSAGE(
            num_features=num_node_features,
            num_classes=num_classes,
            device=args.device,
            init=False
        ).from_hyper_parameter(model_hp).model
    
    model.to(args.device)
    return model 

def test_from_data(model_name, clean_data, perturbed_data, setting='poisoning'):
    model_hp, _ = get_encoder_decoder_hp(model_name)
    num_node_features, num_classes = clean_data.x.size(1), max(clean_data.y).item()+1
    clean_accs = []
    ptb_accs = []
    for seed in tqdm(range(args.repeat)):
        set_seed(seed)

        if setting == 'evasion':
            model = init_model(model_name, model_hp, num_node_features, num_classes)
            train(model, clean_data, args)
            acc0 = test(model, perturbed_data, perturbed_data.test_mask)
            acc1 = test(model, perturbed_data, perturbed_data.test_mask)


        elif setting == 'poisoning':
            model = init_model(model_name, model_hp, num_node_features, num_classes)
            train(model, clean_data, args)
            acc0 = test(model, perturbed_data, perturbed_data.test_mask)

            model = init_model(model_name, model_hp, num_node_features, num_classes)
            train(model, perturbed_data, args)
            acc1 = test(model, perturbed_data, perturbed_data.test_mask)

        clean_accs.append(acc0)
        ptb_accs.append(acc1)

    print('Clean {:.4f} ~ {:.4f}'.format(np.mean(clean_accs), np.std(clean_accs)))
    print('Perturb {:.4f} ~ {:.4f}'.format(np.mean(ptb_accs), np.std(ptb_accs)))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg robust model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epoch', type=int, default=200)

    args = parser.parse_args()

    # seed = 100
    # dataset = Planetoid(os.path.expanduser('~/.cache-autogl'), args.dataset, transform=T.NormalizeFeatures())
    # data = dataset[0].to(args.device)

    
    from torch_geometric.utils import to_dense_adj
    from deeprobust.graph.data import PrePtbDataset,Dataset,Dpr2Pyg

    dpr_data = Dataset(root='/tmp/', name=args.dataset.lower(), setting='prognn')
    dataset = Dpr2Pyg(dpr_data)
    data = dataset.data.to(args.device)
    
    # load perturb data with attacker=mettack
    perturbed_data = PrePtbDataset(root='/tmp/',
            name=args.dataset.lower(),
            attack_method='meta',
            ptb_rate=0.05)
    perturbed_adj = perturbed_data.adj
    perturbed_data = data.clone()
    perturbed_data.edge_index = torch.LongTensor(perturbed_adj.nonzero()).to(args.device)

    print('check attack prop:',(perturbed_adj.todense()!=to_dense_adj(data.edge_index)[0].cpu().numpy()).sum() / len(data.edge_index[0]))
    
    test_from_data(args.model, data, perturbed_data, setting='poisoning')

    
