from tqdm import tqdm
from autogl.datasets import build_dataset_from_name
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from autogl.datasets.utils.conversion import to_dgl_dataset
from helper import get_encoder_decoder_hp


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
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

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
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+valid_size:]], neg_v[neg_eids[test_size+valid_size:]]

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
            "gin",
            "topk"
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument("--device", default="cuda", type=str, help="GPU device")

    args = parser.parse_args()

    dataset = build_dataset_from_name(args.dataset.lower())
    dataset = to_dgl_dataset(dataset)

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

        graph = dataset[0].to(args.device)
        num_features = graph.ndata['feat'].size(1)

        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        trainer = LinkPredictionTrainer(
            model = args.model,
            num_features = num_features,
            lr = 1e-2,
            max_epoch = 100,
            early_stopping_round = 101,
            weight_decay = 0.0,
            device = "auto",
            init = True,
            feval = [Auc],
            loss = "binary_cross_entropy_with_logits",
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {},
                "encoder": model_hp,
                "decoder": decoder_hp
            },
            restricted=False
        )

        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_train_test(graph.cpu())

        dataset_splitted = {
            'train': train_g.to(args.device),
            'train_pos': train_pos_g.to(args.device),
            'train_neg': train_neg_g.to(args.device),
            'test_pos': test_pos_g.to(args.device),
            'test_neg': test_neg_g.to(args.device),
        }

        trainer.train([dataset_splitted], False)
        pre = trainer.evaluate([dataset_splitted], mask="test", feval=Auc)
        result = pre.item()
        res.append(result)

    print(np.mean(res), np.std(res))

"""
AUC 0.8151564430268863
"""
