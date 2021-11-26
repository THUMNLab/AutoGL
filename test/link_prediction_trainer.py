import sys

sys.path.insert(0, "../")

# import autogl.module.train
# import torch_geometric
# exit(0)
#
from autogl.datasets import build_dataset_from_name
# from autogl.solver.classifier.link_predictor import AutoLinkPredictor
from autogl.module.train.evaluation import Auc
import yaml
import random
import torch
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from autogl.module.model.dgl import AutoSAGE


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    # return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes()).edges()
    return neg_src, neg_dst

def negative_sample(data):
    return construct_negative_graph(data, 5)

import autogl.datasets.utils as tmp_utils
tmp_utils.negative_sampling = negative_sample

from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from autogl.module.train.link_prediction_full import LinkPredictionTrainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def split_train_test(g):
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[train_size:]], neg_v[neg_eids[train_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

def split_train_valid_test(g):
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)

    valid_size = int(len(eids) * 0.1)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size - valid_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    valid_pos_u, valid_pos_v =  u[eids[test_size:test_size+valid_size]], v[eids[test_size:test_size+valid_size]]
    train_pos_u, train_pos_v = u[eids[test_size+valid_size:]], v[eids[test_size+valid_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    valid_neg_u, valid_neg_v = neg_u[neg_eids[test_size:test_size+valid_size]], neg_v[neg_eids[test_size:test_size+valid_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[train_size:]], neg_v[neg_eids[train_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size+valid_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    valid_pos_g = dgl.graph((valid_pos_u, valid_pos_v), num_nodes=g.number_of_nodes())
    valid_neg_g = dgl.graph((valid_neg_u, valid_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, valid_pos_g, valid_neg_g, test_pos_g, test_neg_g

if __name__ == "__main__":

    setup_seed(1234)

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        type=str,
        help="dataset to use",
        choices=[
            "Cora",
            "CiteSeer",
            "PubMed",
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", default=0, type=int, help="GPU device")

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    seed = args.seed
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.dataset == 'Cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'CiteSeer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'PubMed':
        dataset = PubmedGraphDataset()

    # configs = yaml.load(open(args.configs, "r").read(), Loader=yaml.FullLoader)
    # configs["hpo"]["name"] = args.hpo
    # configs["hpo"]["max_evals"] = args.max_eval
    # autoClassifier = AutoLinkPredictor.from_config(configs)

    graph = dataset[0].to(args.device)
    num_features = graph.ndata['feat'].size(1)

    autoSAGE = AutoSAGE(
        num_features=num_features,
        num_classes=2,
        device=args.device
    )
    autoSAGE.hyperparams = {
        "num_layers": 3,
        "hidden": [16, 16],
        "dropout": 0.0,
        "act": "relu",
        "agg": "mean",
    }
    autoSAGE.initialize()

    trainer = LinkPredictionTrainer(
        model = autoSAGE,
        num_features = num_features,
        optimizer = None,
        lr = 1e-2,
        max_epoch = 100,
        early_stopping_round = 101,
        weight_decay = 0.0,
        device = "auto",
        init = True,
        feval = [Auc],
        loss = "binary_cross_entropy_with_logits",
    )

    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_train_test(graph.cpu())

    dataset = {
        'train': train_g.to(args.device),
        'train_pos': train_pos_g.to(args.device),
        'train_neg': train_neg_g.to(args.device),
        'test_pos': test_pos_g.to(args.device),
        'test_neg': test_neg_g.to(args.device),
    }

    trainer.train(dataset, True)
    pre = trainer.evaluate(dataset, mask="test", feval=Auc)
    print(pre.item())
    res = trainer.predict(dataset)
    print(res)

    exit(0)

    # train
    autoClassifier.fit(
        dataset,
        time_limit=3600,
        evaluation_method=[Auc],
        seed=seed,
        train_split=0.85,
        val_split=0.05,
    )
    autoClassifier.get_leaderboard().show()

    # test
    predict_result = autoClassifier.predict_proba()

    pos_edge_index, neg_edge_index = (
        dataset[0].test_pos_edge_index,
        dataset[0].test_neg_edge_index,
    )
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E)
    link_labels[: pos_edge_index.size(1)] = 1.0

    print(
        "test auc: %.4f"
        % (Auc.evaluate(predict_result, link_labels.detach().cpu().numpy()))
    )

"""
AUC 0.8151564430268863
"""
