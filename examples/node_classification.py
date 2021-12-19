import yaml
import random
import torch.backends.cudnn
import numpy as np
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module import Acc
from autogl.backend import DependentBackend

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        "auto node classification", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="cora",
        type=str,
        help="dataset to use",
        choices=[
            "cora",
            "pubmed",
            "citeseer",
            "coauthor_cs",
            "coauthor_physics",
            "amazon_computers",
            "amazon_photo",
        ],
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="../configs/nodeclf_gcn_benchmark_small.yml",
        help="config to use",
    )
    # following arguments will override parameters in the config file
    parser.add_argument("--hpo", type=str, default="tpe", help="hpo methods")
    parser.add_argument(
        "--max_eval", type=int, default=50, help="max hpo evaluation times"
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

    dataset = build_dataset_from_name(args.dataset)
    label = dataset[0].nodes.data["y" if DependentBackend.is_pyg() else "label"]
    num_classes = len(np.unique(label.numpy()))

    configs = yaml.load(open(args.configs, "r").read(), Loader=yaml.FullLoader)
    configs["hpo"]["name"] = args.hpo
    configs["hpo"]["max_evals"] = args.max_eval
    autoClassifier = AutoNodeClassifier.from_config(configs)

    # train
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc])
    else:
        autoClassifier.fit(
            dataset,
            time_limit=3600,
            evaluation_method=[Acc],
            seed=seed,
            train_split=20 * num_classes,
            val_split=30 * num_classes,
            balanced=False,
        )
    autoClassifier.get_leaderboard().show()
    acc = autoClassifier.evaluate(metric="acc")
    print("test acc: {:.4f}".format(acc))
