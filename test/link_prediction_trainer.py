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

    trainer = LinkPredictionTrainer(
        model = 'gcn',
        num_features = num_features,
        optimizer = None,
        lr = 1e-4,
        max_epoch = 100,
        early_stopping_round = 101,
        weight_decay = 1e-4,
        device = "auto",
        init = True,
        feval = [Auc],
        loss = "binary_cross_entropy_with_logits",
    )

    dataset = {
        'train_pos': graph,
        'train_neg': graph,
        'val_pos': graph,
        'val_neg': graph,
        'test_pos': graph,
        'test_neg': graph,
    }

    trainer.train(dataset, True)
    pre = trainer.evaluate_dgl(dataset, mask="val", feval=[Auc])
    print(pre)
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
