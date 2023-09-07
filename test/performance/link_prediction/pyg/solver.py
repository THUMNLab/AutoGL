import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import time
from tqdm import tqdm
import numpy as np
from helper import get_encoder_decoder_hp
from autogl.solver import AutoLinkPredictor
from autogl.datasets import build_dataset_from_name

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
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument("--device", default=0, type=int, help="GPU device")

    args = parser.parse_args()

    if args.device < 0:
        device = args.device = "cpu"
    else:
        device = args.device = f"cuda:{args.device}"

    dataset = build_dataset_from_name(args.dataset.lower())

    res = []
    begin_time = time.time()
    for seed in tqdm(range(1234, 1234+args.repeat)):
        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        solver = AutoLinkPredictor(
            feature_module="NormalizeFeatures",
            graph_models=(args.model, ),
            hpo_module="random",
            ensemble_module=None,
            max_evals=1,
            trainer_hp_space=fixed(**{
                "max_epoch": 100,
                "early_stopping_round": 101,
                "lr": 1e-2,
                "weight_decay": 0.0,
            }),
            model_hp_spaces=[{"encoder": fixed(**model_hp), "decoder": fixed(**decoder_hp)}]
        )

        solver.fit(dataset, train_split=0.85, val_split=0.05, evaluation_method=["auc"], seed=seed)
        pre = solver.evaluate(metric="auc")
        res.append(pre)

    print("{:.2f} ~ {:.2f} ({:.2f}s/it)".format(np.mean(res) * 100, np.std(res) * 100, (time.time() - begin_time) / args.repeat))
