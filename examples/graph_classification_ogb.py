import os
os.environ["AUTOGL_BACKEND"] = "pyg"
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
        default="ogbg-molhiv",
        type=str,
        help="graph classification dataset"
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
    # set the mask according to OGB
    split_idx = dataset.get_idx_split()
    train_index = split_idx['train'].tolist()
    val_index = split_idx['valid'].tolist()
    test_index = split_idx['test'].tolist()
    dataset.train_index = train_index
    dataset.val_index = val_index
    dataset.test_index = test_index
    dataset.train_split = [dataset[i] for i in train_index]
    dataset.val_split = [dataset[i] for i in val_index]
    dataset.test_split = [dataset[i] for i in test_index]
    dataset.data.y = dataset.data.y.squeeze(-1)

    autoClassifier = AutoGraphClassifier.from_config(args.configs)

    # train
    autoClassifier.fit(dataset, evaluation_method=[Acc], seed=args.seed)
    autoClassifier.get_leaderboard().show()

    print("best single model:\n", autoClassifier.get_leaderboard().get_best_model(0))

    # test
    acc = autoClassifier.evaluate(metric="acc")
    print("test acc {:.4f}".format(acc))
