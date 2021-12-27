import os
os.environ["AUTOGL_BACKEND"] = "pyg"
from tqdm import tqdm
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
import torch
import numpy as np
import scipy.sparse as sp
from helper import get_encoder_decoder_hp
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling

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

def split_train_test(data):
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
    dataset_splitted = Data(
        x=data.x,train_pos_edge_index=data.train_pos_edge_index,train_neg_edge_index=neg_edge_index,
        test_pos_edge_index=data.test_pos_edge_index,
        test_neg_edge_index = data.test_neg_edge_index, 
        val_pos_edge_index = data.val_pos_edge_index, 
        val_neg_edge_index = data.val_neg_edge_index
        )
    return dataset_splitted


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

    args.dataset = 'Cora'
    args.model = 'gcn'

    path = osp.join('data', args.dataset)
    if args.dataset == 'Cora':
        dataset = Planetoid(path, name='Cora',transform=T.NormalizeFeatures())
    elif args.dataset == 'CiteSeer':
        dataset = Planetoid(path, name='CiteSeer',transform=T.NormalizeFeatures())
    elif args.dataset == 'PubMed':
        dataset = Planetoid(path, name='PubMed',transform=T.NormalizeFeatures())
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

        data = dataset[0].to(args.device)
        num_features = dataset.num_features

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

        dataset_splitted = split_train_test(data.cpu())
        
        trainer.train([dataset_splitted], False)
        pre = trainer.evaluate([dataset_splitted], mask="test", feval=Auc)
        result = pre.item()
        res.append(result)

    print(np.mean(res), np.std(res))