from tqdm import tqdm
from autogl.datasets import build_dataset_from_name
from autogl.solver.classifier.link_predictor import AutoLinkPredictor
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from autogl.module.model.dgl import AutoSAGE, AutoGAT, AutoGCN
from autogl.datasets.utils.conversion import to_dgl_dataset


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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

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
    parser.add_argument(
        "--model",
        default="sage",
        type=str,
        help="model to use",
        choices=[
            "gcn",
            "gat",
            "sage",
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument("--device", default="cuda", type=str, help="GPU device")

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    dataset = build_dataset_from_name(args.dataset.lower())
    dataset = to_dgl_dataset(dataset)
    train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = split_train_valid_test(dataset[0].cpu())

    dataset = [[train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g]]

    res = []
    for seed in tqdm(range(1234, 1234+args.repeat)):
        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.model == 'sage':
            model_hp = {
                "num_layers": 3,
                "hidden": [16, 16],
                "dropout": 0.0,
                "act": "relu",
                "agg": "mean",
            }
        else:
            model_hp = dict()

        autoClassifier = AutoLinkPredictor(
            feature_module=None,
            graph_models=(args.model,),
            ensemble_module=None,
            max_evals=1,
            hpo_module='random',
            trainer_hp_space=fixed(**{
                "max_epoch": 100,
                "early_stopping_round": 100 + 1,
                "lr":0.01,
                "weight_decay": 0.0,
            }),
            model_hp_spaces=[fixed(**model_hp)]
        )
        autoClassifier.fit(
            dataset,
            time_limit=3600,
            evaluation_method=[Auc],
            seed=seed,
            # train_split=0.85,
            # val_split=0.05,
        )
        autoClassifier.get_leaderboard().show()

        # test
        predict_result = autoClassifier.predict_proba()

        pos_edge_index, neg_edge_index = torch.stack(dataset[0][-2].edges()), torch.stack(dataset[0][-1].edges())
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E)
        link_labels[: pos_edge_index.size(1)] = 1.0

        print(
            "test auc: %.4f"
            % (Auc.evaluate(predict_result, link_labels.detach().cpu().numpy()))
        )
