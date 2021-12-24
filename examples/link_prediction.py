from autogl.datasets import build_dataset_from_name
from autogl.solver.classifier.link_predictor import AutoLinkPredictor
from autogl.module.train.evaluation import Auc
import yaml
import random
import torch
import numpy as np

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
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
        default="../configs/lp_gcn_benchmark.yml",
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

    configs = yaml.load(open(args.configs, "r").read(), Loader=yaml.FullLoader)
    configs["hpo"]["name"] = args.hpo
    configs["hpo"]["max_evals"] = args.max_eval
    autoClassifier = AutoLinkPredictor.from_config(configs)

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

    auc = autoClassifier.evaluate(metric="auc")
    print("test auc: {:.4f}".format(auc))
