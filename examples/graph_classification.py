"""
Example of graph classification on given datasets.
This version use random split to only show the usage of AutoGraphClassifier.
Refer to `graph_cv.py` for cross validation evaluation of the whole system
following paper `A Fair Comparison of Graph Neural Networks for Graph Classification`
"""

import sys

sys.path.append("../")
import random
import torch
import numpy as np
from autogl.datasets import build_dataset_from_name, utils
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
        "--configs", default="../configs/graphclf_full.yml", help="config files"
    )
    parser.add_argument("--device", type=int, default=0, help="device to run on")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

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

    dataset = build_dataset_from_name(args.dataset)
    if args.dataset.startswith("imdb"):
        from autogl.module.feature.generators import PYGOneHotDegree

        # get max degree
        from torch_geometric.utils import degree

        max_degree = 0
        for data in dataset:
            deg_max = int(degree(data.edge_index[0], data.num_nodes).max().item())
            max_degree = max(max_degree, deg_max)
        dataset = PYGOneHotDegree(max_degree).fit_transform(dataset, inplace=False)
    elif args.dataset == "collab":
        from autogl.module.feature.auto_feature import Onlyconst

        dataset = Onlyconst().fit_transform(dataset, inplace=False)
    utils.graph_random_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=args.seed)

    autoClassifier = AutoGraphClassifier.from_config(args.configs)

    # train
    autoClassifier.fit(dataset, evaluation_method=[Acc], seed=args.seed)
    autoClassifier.get_leaderboard().show()

    print("best single model:\n", autoClassifier.get_leaderboard().get_best_model(0))

    # test
    predict_result = autoClassifier.predict_proba()
    print(
        "test acc %.4f"
        % (
            Acc.evaluate(
                predict_result,
                dataset.data.y[dataset.test_index].cpu().detach().numpy(),
            )
        )
    )
