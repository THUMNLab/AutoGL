import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import time

sys.path.append("../../")
os.environ["AUTOGL_BACKEND"] = "dgl"
# os.environ["AUTOGL_BACKEND"] = "pyg"
from autogl.backend import DependentBackend
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, GINDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from autogl.module.model import GAT,GraphSAGE,AutoSAGE,AutoGCN,AutoGAT

from pdb import set_trace
import numpy as np
from autogl.solver.utils import set_seed
set_seed(202106)
import argparse

def evaluate(model, graph, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main():
    

    # set up seeds, args.seed supported
    torch.manual_seed(seed=202106)
    np.random.seed(seed=202106)

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed=202106)
    else:
        device = torch.device("cpu")

    dataset = CoraGraphDataset()
    data = dataset[0].to(device)
    data.ndata['x'] = data.ndata['feat']
    train_mask = data.ndata['train_mask']
    val_mask = data.ndata['val_mask']
    test_mask = data.ndata['test_mask']
    labels = data.ndata['label']
    n_edges = data.number_of_edges()

    # args={}
    # args["features_num"]=data.ndata['x'].size(1)
    # args['hidden']=[16]
    # args["heads"]=8
    # args['dropout']=0.6
    # args["num_class"]=dataset.num_classes
    # args["num_layers"]=2
    # args['act']='relu'


    # model = GAT(args)
    # model = GraphSAGE(args["features_num"],
    #                   args["num_class"],
    #                   [16],'relu',0.5)
    automodel = AutoGAT(
        num_features = data.ndata['x'].size(1),
        num_classes = dataset.num_classes,
        device = device,
        init = True
    )

    model = automodel.model

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dur = []
    for epoch in range(200):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(data)
        loss = criterion(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, data, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, data, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    
    main()

