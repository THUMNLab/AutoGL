from torch_geometric.nn import GCNConv, SAGEConv
from nni.nas.pytorch import mutables
import torch.nn as nn


class BaseNAS:
    def to(self, device):
        """
        change the device of the whole process
        """
        self.device = device

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

class BaseEstimator:
    def __init__(self, device="cuda"):
        self.device = device

    def infer(self, model, dataset):
        pass

class DartsNodeClfEstimator(BaseEstimator):
    def infer(self, model, dataset):
        dset = dataset[0].to(self.device)
        pred = model(dset)[dset.train_mask]
        y = dset.y[dset.train_mask]
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred, y)
        return loss, loss
