import torch
from deeprobust.graph.global_attack import Random,DICE
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import time
import torch.nn.functional as F

from . import register_nas_estimator
from ..space import BaseSpace
from .base import BaseEstimator
from ..backend import *
from ...train.evaluation import Acc
from ..utils import get_hardware_aware_metric

@register_nas_estimator("grna")
class GRNAEstimator(BaseEstimator):
    """
    Graph robust neural architecture estimator under adversarial attack.

    Use model directly to get estimations.

    Parameters
    ----------
    loss_f : str
        The name of loss funciton in PyTorch
    evaluation : list of Evaluation
        The evaluation metrics in module/train/evaluation
        GRNA_metric = acc_metric+ robustness_metric
    lambda: float
        The hyper-parameter to balance the accuracy metric and robustness metric to perform ultimate evaluation
    perturb_type: str
        Perturbation methods to simulate the adversarial attack process
    adv_sample_num: int
        Adversarial sample number used in measure architecture robustness.
    """

    def __init__(self, 
                 loss_f="nll_loss", 
                 evaluation=[Acc()], 
                 lambda_=0.05, 
                 perturb_type='random',
                 adv_sample_num=10,  
                 dis_type='ce',
                 ptbr=0.05):
        super().__init__(loss_f, evaluation)
        self.evaluation = evaluation
        self.lambda_ = lambda_
        self.perturb_type = perturb_type
        self.dis_type = dis_type
        self.adv_sample_num = adv_sample_num
        self.ptbr = ptbr
        print('initialize GRNA estimator')

    def infer(self, model: BaseSpace, dataset, mask="train"):
        
        device = next(model.parameters()).device
        dset = dataset[0].to(device)
        mask = bk_mask(dset, mask)

        pred = model(dset)[mask]
        label = bk_label(dset)
        y = label[mask]

        loss = getattr(F, self.loss_f)(pred, y)
        probs = F.softmax(pred, dim=1).detach().cpu().numpy()

        # robustness metric
        dist = 0
        for _ in range(self.adv_sample_num):
            modified_adj = self.gen_adversarial_samples(dset.edge_index, num_nodes=dset.num_nodes, perturb_prop=self.ptbr, attack_method=self.perturb_type)
            d_data = dset.clone()
            d_data = d_data.to(device)
            edge_index = torch.LongTensor(np.vstack((modified_adj.tocoo().row,modified_adj.tocoo().col)))
            d_data.edge_index = edge_index.to(device)
            perturb_pred = model(d_data)[mask]
            dist += distance(perturb_pred, pred, dis_type=self.dis_type)
        dist = dist/self.adv_sample_num
        
        y = y.cpu()
        metrics = [eva.evaluate(probs, y)+self.lambda_*dist for eva in self.evaluation]

        return metrics, loss


    def gen_adversarial_samples(self, edge_index, num_nodes, perturb_prop, attack_method='random'):
    
        if num_nodes is None:
            num_nodes = max(edge_index[0])+1
        
        edge_index = edge_index.detach().cpu().numpy()
        delta = int(edge_index.shape[1]//2 * perturb_prop)
        v = np.ones_like(edge_index[0])
        adj = sp.csr_matrix((v,(edge_index[0], edge_index[1])), shape=(num_nodes,num_nodes))
        if attack_method=='random':
            attacker = Random()
            attacker.attack(adj, n_perturbations=delta, type='flip')
        elif attack_method=='dice':
            labels = self.data.y.cpu().numpy()
            attacker = DICE()
            attacker.attack(adj, labels, delta)
        else:
            assert False, 'Wrong Type of attack method!'

        modified_adj = attacker.modified_adj  # scipy.sparse matrix
        
        return modified_adj.tocsr()




def distance(perturb, clean, dis_type='ce', data=None, p=2):
    """
    Distance between logits of perturbed and clean data.
    Parameters:
    ---------
    perturb: torch.Tensor [n, C]
    clean: torch.Tensor [n,C]
    type: loss type
    labels: ground truth labels, needed when type='cw'
    p: fro norm, needed when type='fro'

    Return
    ------
    Distance: torch.Tensor [n,]
    """
    # if type=='cos':
    #     return perturb*clean / torch.sqrt(torch.norm(perturb,p=2) * torch.norm(clean))
    if dis_type=='fro':
        return torch.norm(perturb - clean,p=p)

    elif dis_type=='ce':
        p_ = F.softmax(clean,-1)
        logq = F.log_softmax(perturb, -1)
        return -(p_*logq).mean()

    elif dis_type=='kl':
        logq = F.log_softmax(perturb,-1)
        p_ = F.softmax(clean, -1)
        logp = F.log_softmax(clean,-1)
        return (p_*(logp-logq)).mean()*100

    elif dis_type=='cw':
        perturb, clean, labels = perturb[data.train_mask],clean[data.train_mask], data.y[data.train_mask]
        eye = torch.eye(labels.max() + 1)
        onehot_mx = eye[labels]
        one_hot_labels =  onehot_mx.to(labels.device)
        # perturb
        ptb_best_second_class = (perturb - 1000*one_hot_labels).argmax(1)
        margin = perturb[np.arange(len(perturb)), labels]  - \
            perturb[np.arange(len(perturb)), ptb_best_second_class]
        ptb_loss = -torch.clamp(margin, max = 0, min = -0.1).mean()
        # clean
        clean_best_second_class = (perturb - 1000*one_hot_labels).argmax(1)
        margin = clean[np.arange(len(perturb)), labels]  - \
            clean[np.arange(len(perturb)), clean_best_second_class]
        clean_loss = -torch.clamp(margin, max = 0, min = -0.1).mean()

        return (ptb_loss-clean_loss)*100