"""
Auto graph classification using cross validation methods proposed in
paper `A Fair Comparison of Graph Neural Networks for Graph Classification`
"""

import sys
import random
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append("../")

from autogl.datasets import build_dataset_from_name, utils
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc
from autogl.backend import DependentBackend
if DependentBackend.is_pyg():
    from autogl.datasets.utils.conversion import to_pyg_dataset as convert_dataset
else:
    from autogl.datasets.utils.conversion import to_dgl_dataset as convert_dataset

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
    parser.add_argument("--folds", type=int, default=10, help="fold number")

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

    print("begin processing dataset", args.dataset, "into", args.folds, "folds.")
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
    utils.graph_cross_validation(dataset, args.folds, random_seed=args.seed)

    accs = []
    for fold in range(args.folds):
        print("evaluating on fold number:", fold)
        utils.set_fold(dataset, fold)
        train_dataset = utils.graph_get_split(dataset, "train", False)
        autoClassifier = AutoGraphClassifier.from_config(args.configs)

        autoClassifier.fit(
            train_dataset,
            train_split=0.9,
            val_split=0.1,
            seed=args.seed,
            evaluation_method=[Acc],
        )
        acc = autoClassifier.evaluate(dataset, mask="val", metric="acc")
        print("test acc fold {:d}: {:.4f}".format(fold, acc))
        accs.append(acc)
    print("Average acc on", args.dataset, ":", np.mean(accs), "~", np.std(accs))
