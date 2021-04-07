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
        "--configs", default="../configs/graph_classification.yaml", help="config files"
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
    utils.graph_cross_validation(dataset, args.folds, random_seed=args.seed)

    accs = []
    for fold in range(args.folds):
        print("evaluating on fold number:", fold)
        utils.graph_set_fold_id(dataset, fold)
        train_dataset = utils.graph_get_split(dataset, "train", False)
        autoClassifier = AutoGraphClassifier.from_config(args.configs)

        autoClassifier.fit(
            train_dataset,
            train_split=0.9,
            val_split=0.1,
            seed=args.seed,
            evaluation_method=[Acc],
        )
        predict_result = autoClassifier.predict_proba(dataset, mask="val")
        acc = Acc.evaluate(
            predict_result, dataset.data.y[dataset.val_index].cpu().detach().numpy()
        )
        print(
            "test acc %.4f"
            % (
                Acc.evaluate(
                    predict_result,
                    dataset.data.y[dataset.val_index].cpu().detach().numpy(),
                )
            )
        )
        accs.append(acc)
    print("Average acc on", args.dataset, ":", np.mean(accs), "~", np.std(accs))
