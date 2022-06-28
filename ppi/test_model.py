"""
Performance check of AutoGL model + PYG (trainer + dataset)
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

os.environ["AUTOGL_BACKEND"] = "pyg"

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from autogl.module.model.pyg import AutoGCN, AutoGAT, AutoSAGE, AutoGATPooling
from autogl.datasets import utils
from autogl.solver.utils import set_seed
from helper import get_encoder_decoder_hp
import logging
from datetime import datetime
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from utils import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import (accuracy_score)



def train(model,train_loader,optimizer,device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.batch)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model,loader,device):
    model.eval()
    ys = []
    preds = []
    for data in loader:
        ys.append(data.y)
        data = data.to(device)
        out = model(data.x, data.batch).cpu()
            
        preds.append(out)
        # print("edge_index:",edge_index, alpha)

    ys = torch.cat(ys).numpy()
    preds = F.softmax(torch.cat(preds), dim = 1).numpy()
    preds = np.argmax(preds, axis=1)
    acc=accuracy_score(ys,preds)

    return acc


def get_model_att(test_dataset, model):
    """output attentions for test_dataset of current model
    """

    all_atts = []
    for data in test_dataset:
        data = data.to(device)
        data.batch = torch.LongTensor([0]*data.num_nodes).to(device)
        _, attentions = model(data.x, data.batch, return_attention_weights=True)
        all_atts.append(attentions)
    return all_atts

def train_till_end(fold,device,foldi, args, metadata):
    train_loader=DataLoader(fold[0], args.batch_size, shuffle = True)
    val_loader=DataLoader(fold[1], args.batch_size)
    test_loader=DataLoader(fold[2], args.batch_size)
    test_dataset = DataLoader(fold[2], 1)
    model = AutoGATPooling(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            device=device,
            init=False,
            num_nodes=num_nodes
        ).from_hyper_parameter(
            {
            "hidden": [16,16,16],
            "dropout": 0.,
            "act": "relu",
            "heads":4,
            "metadata":metadata,
        }
        ).model
    
    model.to(device)
        # model = Net(num_features,hid_dim,num_classes,metadata).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    earlystop = EarlyStopping(mode="max", patience=patience)
    best_val=0
    best_test=0
    best_attentions = None
    with tqdm(range(args.max_epochs)) as bar:
        for epoch in bar:
            loss = train(model,train_loader,optimizer,device)
            # train_metric=test(model,train_loader,device)
            train_metric=0
            if val_ratio>0:
                val_metric=test(model,val_loader,device,)
            else:
                val_metric=test(model,train_loader,device)
            test_metric=test(model,test_loader,device)

            if val_metric>best_val:
                best_val=val_metric
                best_test=test_metric
                best_attentions=get_model_att(test_dataset, model)
            bar.set_postfix(loss=loss,btest=best_test,bval=best_val,test_metric=test_metric,val_metric=val_metric,train_metric=train_metric)
            
            if writer:
                writer.add_scalar(f"Model{foldi}/train_loss", loss, epoch)
                writer.add_scalar(f"Model{foldi}/train_metric", train_metric, epoch)
                writer.add_scalar(f"Model{foldi}/test_metric", test_metric, epoch)
                writer.add_scalar(f"Model{foldi}/val_metric", val_metric, epoch)
            
            if earlystop.step(val_metric):
                break
    print(f'Acc :{best_test}')

    return best_test, best_attentions


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('pyg model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=500)

    args = parser.parse_args()

    device=torch.device('cuda')
    logging.basicConfig(level=logging.ERROR)

    seed_everything(0)
    log_dir='logs/'
    timestamp = datetime.now().strftime('%m%d-%H-%M-%S')
    writer = SummaryWriter(log_dir)

    # read data
    datafile='./data/processed2-PPI-3'
    data=torch.load(datafile) # x y ei ef inner_links cross_links hid2ids
    x=data['x']
    y=data['y']
    edge_index=data['ei'].to(device)
    # edge_weight=data['ef']
    inner_edge_index=data['inner_links']
    cross_edge_index=data['cross_links']
    inner_edge_index=[i.to(device) for i in inner_edge_index]
    cross_edge_index=[i.to(device) for i in inner_edge_index]

    num_nodes=[len(_) for _ in data['hid2ids']] # num of nodes from 1 to last layer
    metadata=[edge_index,inner_edge_index,cross_edge_index,num_nodes]

    datas=[Data(x=x[i],y=int(y[i])) for i in range(x.shape[0])]

    train_dataset=dataset=MPPIDataset(datas)
    num_features=dataset.num_features
    num_classes=dataset.num_classes

    #### hyper
    # device='cpu'
    
    # hid_dim=16
    # max_epochs = 500
    # batch_size = 32 #32
    batch_size = 32
    patience = 50
    val_ratio=0
    # lr=1e-4
    # lr = 5e-5
    #### folds
    folds=get_fold(datas,5,val_ratio=val_ratio)
    print('folds:',folds)

    accs = []

    for seed in tqdm(range(args.repeat)):
        set_seed(seed)
        results = []
        for i,fold in enumerate(folds):
            f1,best_attentions=train_till_end(fold,device,i, args,metadata)
            results.append(f1)
        
        accs.append(np.mean(results))
        


    print('{:.4f} ~ {:.4f}'.format(np.mean(accs), np.std(accs)))
