import numpy as np
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoHeteroNodeClassifier
from helper import get_encoder_decoder_hp
from tqdm import tqdm

def fixed(**kwargs):
    return [{
        'parameterName': k,
        "type": "FIXED",
        "value": v
    } for k, v in kwargs.items()]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["han", "hgt", "HeteroRGCN"], default="hgt")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeat", type=int, default=10)

    args = parser.parse_args()

    dataset = {
        "han": "hetero-acm-han",
        "hgt": "hetero-acm-hgt",
        "HeteroRGCN": "hetero-acm-hgt"
    }

    dataset = build_dataset_from_name(dataset[args.model])

    model_hp, _ = get_encoder_decoder_hp(args.model)

    accs = []
    process = tqdm(total=args.repeat)
    for rep in range(args.repeat):
        solver = AutoHeteroNodeClassifier(
            graph_models=[args.model],
            hpo_module="random",
            ensemble_module=None,
            max_evals=1,
            device=args.device,
            trainer_hp_space=fixed(
                max_epoch=args.epoch,
                early_stopping_round=args.epoch + 1,
                lr=args.lr,
                weight_decay=args.weight_decay
            ),
            model_hp_spaces=[fixed(**model_hp)]
        )
        solver.fit(dataset)
        acc = solver.evaluate()
        accs.append(acc)
        process.update(1)
        process.set_postfix(mean=np.mean(accs), std=np.std(accs))
    process.close()
    print("mean: {:.4f} ~ std: {:.4f}".format(np.mean(accs), np.std(accs)))
