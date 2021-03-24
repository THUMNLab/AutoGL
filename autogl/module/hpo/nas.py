from torch_geometric.nn import GCNConv, SAGEConv
from nni.nas.pytorch import mutables
import torch.nn as nn

class BaseNAS:
    def search(self, space, dset, trainer):
        """
        The main process of NAS.
        Parameters
        ----------
        space : BaseArchitectureSpace
            No implementation yet
        dataset : ...datasets
            Dataset to train and evaluate.
        trainer : ..train.BaseTrainer
            Including model, giving HP space and using for training

        Returns
        -------
        model: ..train.BaseTrainer
            The trainer including the best trained model
        """

class GraphSpace(nn.Module):
    def __init__(self, inp, hid, oup):
        super().__init__()
        self.gcn = GCNConv(inp, hid)
        self.op1 = mutables.LayerChoice([GCNConv(inp, hid),SAGEConv(inp, hid)])
        self.op2 = mutables.LayerChoice([
            GCNConv(hid, oup),
            SAGEConv(hid, oup)       
        ], key = "2")

    def forward(self, data):
        x = self.op1(data.x, data.edge_index)
        x = self.op2(x, data.edge_index)
        return x
        
class BaseTrainer:
    def infer(self, model, dataset):
        dset = dataset[0]
        pred = model(dset)[dset.train_mask]
        y = dset.y[dset.train_mask]
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred, y)
        return loss, loss