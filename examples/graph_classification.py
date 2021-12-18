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
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
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
        from autogl.module.feature import OneHotDegreeGenerator

        # get max degree
        from autogl.module.feature._generators._pyg_impl import degree

        max_degree = 0
        for data in dataset:
            deg_max = int(degree(data.edge_index[0], data.num_nodes).max().item())
            max_degree = max(max_degree, deg_max)
        dataset = OneHotDegreeGenerator(max_degree).fit_transform(dataset, inplace=False)
    elif args.dataset == "collab":
        # FIXME: no onlyconst feature engineer ??
        # FIXME: no auto feature engineer support !!
        from autogl.module._feature.auto_feature import Onlyconst

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
                np.array([d.data["y" if backend == "pyg" else "label"] for d in dataset.test_split]),
            )
        )
    )
