from tqdm import tqdm
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from helper import get_encoder_decoder_hp

from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from autogl.module.train.link_prediction_full import LinkPredictionTrainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

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
        default="PubMed",
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
        default="gat",
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

    if args.dataset == 'Cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'CiteSeer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'PubMed':
        dataset = PubmedGraphDataset()
    else:
        assert False

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

        graph = dataset[0]
        num_features = graph.ndata['feat'].size(1)

        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        trainer = LinkPredictionTrainer(
            model = args.model,
            num_features = num_features,
            lr = 1e-2,
            max_epoch = 100,
            early_stopping_round = 101,
            weight_decay = 0.0,
            device = args.device,
            feval = [Auc],
            loss = "binary_cross_entropy_with_logits",
            init = False
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {},
                "encoder": model_hp,
                "decoder": decoder_hp
            },
            restricted=False
        )

        gs = list(split_train_valid_test(graph))
    
        if args.model == 'gcn' or args.model == 'gat':
            gs[0] = dgl.add_self_loop(gs[0])

        dataset_splitted = {
            'train': gs[0].to(args.device),
            'train_pos': gs[1].to(args.device),
            'train_neg': gs[2].to(args.device),
            'val_pos': gs[3].to(args.device),
            'val_neg': gs[4].to(args.device),
            'test_pos': gs[5].to(args.device),
            'test_neg': gs[6].to(args.device),
        }

        trainer.train([dataset_splitted], True)
        pre = trainer.evaluate([dataset_splitted], mask="test", feval=Auc)
        result = pre
        res.append(result)

    print("{:.2f} ~ {:.2f}".format(np.mean(res) * 100, np.std(res) * 100))
