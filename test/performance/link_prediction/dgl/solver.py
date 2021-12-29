import torch
import random
import numpy as np

import dgl
from tqdm import tqdm
from autogl.datasets import build_dataset_from_name
from autogl.solver.classifier.link_predictor import AutoLinkPredictor
from autogl.datasets.utils.conversion import to_dgl_dataset
from autogl.datasets.utils import split_edges
from helper import get_encoder_decoder_hp

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

if __name__ == "__main__":


    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        type=str,
        help="dataset to use",
        choices=[
            "Cora",
            "CiteSeer",
            "PubMed",
        ],
    )
    parser.add_argument(
        "--model",
        default="sage",
        type=str,
        help="model to use",
        choices=[
            "gcn",
            "gat",
            "sage",
            "gin",
            "topk"
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument("--device", default="cuda", type=str, help="GPU device")

    args = parser.parse_args()

    dataset = build_dataset_from_name(args.dataset.lower())
    dataset = to_dgl_dataset(dataset)

    res = []
    for seed in tqdm(range(1234, 1234+args.repeat)):
        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        gs = list(split_edges(dataset, 0.8, 0.1)[0])

        if args.model == 'gcn' or args.model == 'gat':
            gs[0] = dgl.add_self_loop(gs[0])

        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        autoClassifier = AutoLinkPredictor(
            feature_module=None,
            graph_models=(args.model,),
            ensemble_module=None,
            max_evals=1,
            hpo_module='random',
            trainer_hp_space=fixed(**{
                "max_epoch": 100,
                "early_stopping_round": 100 + 1,
                "lr":0.01,
                "weight_decay": 0.0,
            }),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}],
            device=args.device
        )
        autoClassifier.fit(
            [gs],
            time_limit=3600,
            evaluation_method=["auc"],
            seed=seed,
        )
        auc = autoClassifier.evaluate(metric='auc')
        res.append(auc)

print("{:.2f} ~ {:.2f}".format(np.mean(res) * 100, np.std(res) * 100))
