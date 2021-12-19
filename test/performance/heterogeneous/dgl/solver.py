from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoHeteroNodeClassifier

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["han", "hgt", "heteroRGCN"], default="hgt")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeat", type=int, default=10)

    args = parser.parse_args()

    dataset = {
        "han": "hetero-acm-han",
        "hgt": "hetero-acm-hgt",
        "heteroRGCN": "hetero-acm-hgt"
    }

    dataset = build_dataset_from_name(dataset[args.model])

    for rep in range(args.repeat):
        pass
