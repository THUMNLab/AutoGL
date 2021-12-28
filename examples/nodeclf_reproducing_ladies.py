import sys
import yaml
import random
import torch.backends.cudnn
import numpy as np
sys.path.append("../")
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module import MicroF1


torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    import argparse
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset",
        default="pubmed",
        type=str,
        help="dataset to use",
        choices=[
            "cora",
            "pubmed",
            "citeseer",
            "reddit"
        ],
    )
    argument_parser.add_argument(
        "--configs",
        type=str,
        default="../configs/nodeclf_ladies_reproduction.yml",
        help="configuration file to adopt",
    )
    argument_parser.add_argument("--seed", type=int, default=0, help="random seed")
    argument_parser.add_argument("--device", default=0, type=int, help="GPU device")

    arguments = argument_parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(arguments.device)
    seed = arguments.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = build_dataset_from_name(arguments.dataset)
    configs = yaml.load(
        open(arguments.configs, "r").read(),
        Loader=yaml.FullLoader
    )
    autoClassifier = AutoNodeClassifier.from_config(configs)
    # The running time is likely to exceed 1 hour when CiteSeer or Reddit dataset is adopted
    autoClassifier.fit(dataset, time_limit=24 * 3600, evaluation_method=[MicroF1])
    autoClassifier.get_leaderboard().show()
    predict_result = autoClassifier.predict_proba()
    res = autoClassifier.evaluate(metric=[MicroF1, 'acc'])
    print("Final Micro F1 {:.4f} Acc {:.4f}".format(res[0], res[1]))
