import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import DataLoader

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from ..backend import *
from ...train.evaluation import Acc
from ..utils import get_hardware_aware_metric

@register_nas_estimator("oneshot")
class OneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, mask="train"):
        # print("mask",mask)
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)

        pred = model(dset)[mask]
        label = bk_label(dset)
        y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()
        
        y = y.cpu()
        metrics = [eva.evaluate(probs, y) for eva in self.evaluation]
        return metrics, loss

@register_nas_estimator("gcloneshot")
class GCLOneShotEstimator(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, mask="train"):
        device = next(model.parameters()).device
        out_auc = 0
        out_loss = 0
        # print("one_1")
        for i, fold in enumerate(dataset):
            print("~~~~~~~~~~~~~~fold",i)
            if mask == "train":
                loader=DataLoader(fold[0], 1, shuffle = True) # train_loader 1: batch_size

            elif mask == "test":
                loader=DataLoader(fold[1], 1, shuffle = True) # test_loader
        
            ys = []
            preds = []
            total_loss = 0
            for data in loader:
                # print("one_2")
                ys.append(data.y)
                data = data.to(device)
                # out = model(data.x, data.edge_index, data.batch).cpu()
                out = model(data)
                # print(out)
                # print("one_3")
                # out = a.squeeze(0)
                # out = model(data)[mask].squeeze(0)
                preds.append(out)
                loss = F.nll_loss(out, data.y)
                total_loss += float(loss) * data.num_graphs
                # break # 记得后面去掉
            total_loss = total_loss / len(loader.dataset)
            ys = torch.cat(ys).numpy()
            preds = F.softmax(torch.cat(preds), dim = 1)[:,1].detach().numpy()
            fpr, tpr, th = roc_curve(ys, preds, pos_label = 1)
            out_auc += auc(fpr, tpr)
            out_loss += total_loss

        out_auc = out_auc / float(len(dataset))
        total_loss = total_loss / float(len(dataset))
        return out_auc, total_loss

@register_nas_estimator("gcloneshot2")
class GCLOneShotEstimator2(BaseEstimator):
    """
    One shot estimator.

    Use model directly to get estimations.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    """

    def __init__(self, loss_f="nll_loss", evaluation=[Acc()]):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, batch, mask="train"):
        device = next(model.parameters()).device
        out_auc = 0
        out_loss = 0
        # print("one_1")
        for i, fold in enumerate(dataset):
            print("~~~~~~~~~~~~~~fold",i)
            if mask == "train":
                loader=DataLoader(fold[0], 1, shuffle = True) # train_loader 1: batch_size

            elif mask == "test":
                loader=DataLoader(fold[2], 1, shuffle = True) # test_loader

            elif mask == "val":
                loader=DataLoader(fold[1], 1, shuffle = True) # val_loader，新加的，后面记得改！！
        
            ys = []
            preds = []
            total_loss = 0
            for data in loader:
                # print("one_2")
                ys.append(data.y)
                data = data.to(device)
                # out = model(data.x, data.edge_index, data.batch).cpu()
                out = model(data, batch)
                # print(out)
                # print("one_3")
                # out = a.squeeze(0)
                # out = model(data)[mask].squeeze(0)
                preds.append(out)
                loss = F.nll_loss(out, data.y)
                total_loss += float(loss) * data.num_graphs
                # break # 记得后面去掉
            total_loss = total_loss / len(loader.dataset)
            ys = torch.cat(ys).numpy()
            preds = F.softmax(torch.cat(preds), dim = 1)[:,1].detach().numpy()
            fpr, tpr, th = roc_curve(ys, preds, pos_label = 1)
            out_auc += auc(fpr, tpr)
            out_loss += total_loss

        out_auc = out_auc / float(len(dataset))
        total_loss = total_loss / float(len(dataset))
        return out_auc, total_loss


@register_nas_estimator("oneshot_hardware")
class OneShotEstimator_HardwareAware(OneShotEstimator):
    """
    One shot hardware-aware estimator.

    Use model directly to get estimations with some hardware-aware metrics.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
    hardware_evaluation : str or runable
        The hardware-aware metrics. Can be 'parameter' or 'latency'. Or you can define a special metric by a runable function
    hardware_metric_weight : float
        The weight of hardware-aware metric, which will be a bias added to metrics
    """

    def __init__(
        self,
        loss_f="nll_loss",
        evaluation=[Acc()],
        hardware_evaluation="parameter",
        hardware_metric_weight=0,
    ):
        super().__init__(loss_f, evaluation)
        self.hardware_evaluation = hardware_evaluation
        self.hardware_metric_weight = hardware_metric_weight

    def infer(self, model: BaseSpace, dataset, mask="train"):
        metrics, loss = super().infer(model, dataset, mask)
        if isinstance(self.hardware_evaluation, str):
            hardware_metric = get_hardware_aware_metric(model, self.hardware_evaluation)
        else:
            hardware_metric = self.hardware_evaluation(model)
        metrics = [x - hardware_metric * self.hardware_metric_weight for x in metrics]
        metrics.append(hardware_metric)
        return metrics, loss
