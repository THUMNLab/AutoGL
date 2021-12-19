"""
Example of graph classification on given datasets.
This version use random split to only show the usage of AutoGraphClassifier.
Refer to `graph_cv.py` for cross validation evaluation of the whole system
following paper `A Fair Comparison of Graph Neural Networks for Graph Classification`
"""
import random
import torch
import numpy as np
from autogl.datasets import build_dataset_from_name, utils
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from autogl.backend import DependentBackend
if DependentBackend.is_pyg():
    from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset
else:
    from autogl.datasets.utils.conversion import to_dgl_dataset as convert_dataset

backend = DependentBackend.get_backend_name()

if __name__ == "__main__":
    parser = ArgumentParser(
        "auto graph classification", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="mutag",
        type=str,
        help="graph classification dataset",
        choices=["mutag", "imdb-b", "imdb-m", "proteins", "collab"],
    )
    parser.add_argument(
        "--configs", default="../configs/graphclf_gin_benchmark.yml", help="config files"
    )
    parser.add_argument("--device", type=int, default=-1, help="device to run on, -1 means cpu")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    if args.device == -1:
        args.device = "cpu"

    if torch.cuda.is_available() and args.device != "cpu":
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

    dataset = build_dataset_from_name(args.dataset)
    _converted_dataset = convert_dataset(dataset)
    if args.dataset.startswith("imdb"):
        from autogl.module.feature import OneHotDegreeGenerator

        if DependentBackend.is_pyg():
            from torch_geometric.utils import degree
            max_degree = 0
            for data in _converted_dataset:
                deg_max = int(degree(data.edge_index[0], data.num_nodes).max().item())
                max_degree = max(max_degree, deg_max)
        else:
            max_degree = 0
            for data, _ in _converted_dataset:
                deg_max = data.in_degrees().max().item()
                max_degree = max(max_degree, deg_max)
        dataset = OneHotDegreeGenerator(max_degree).fit_transform(dataset, inplace=False)
    elif args.dataset == "collab":
        from autogl.module.feature._auto_feature import OnlyConstFeature

        dataset = OnlyConstFeature().fit_transform(dataset, inplace=False)
    utils.graph_random_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=args.seed)

    autoClassifier = AutoGraphClassifier.from_config(args.configs)

    # train
    autoClassifier.fit(dataset, evaluation_method=[Acc], seed=args.seed)
    autoClassifier.get_leaderboard().show()

    print("best single model:\n", autoClassifier.get_leaderboard().get_best_model(0))

    # test
    acc = autoClassifier.evaluate(metric="acc")
    print("test acc {:.4f}".format(acc))
