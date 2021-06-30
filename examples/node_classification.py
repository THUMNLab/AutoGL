import sys
import yaml
import random
import torch.backends.cudnn
import numpy as np
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module import Acc
sys.path.append("../")

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
            train_split=20 * dataset.num_classes,
            val_split=30 * dataset.num_classes,
            balanced=False,
        )
    autoClassifier.get_leaderboard().show()

    # test
    predict_result = autoClassifier.predict_proba()
    print(
        "test acc: %.4f"
        % (Acc.evaluate(predict_result, dataset.data.y[dataset.data.test_mask].numpy()))
    )
