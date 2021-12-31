import os
os.environ["AUTOGL_BACKEND"] = 'dgl'

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoHeteroNodeClassifier
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["han", "hgt"])
    parser.add_argument("--max_evals", type=int, default=10)

    args = parser.parse_args()

    dataset = build_dataset_from_name(f"hetero-acm-{args.model}")
    solver = AutoHeteroNodeClassifier(
        graph_models=(args.model, ),
        max_evals=10
    )
    solver.fit(dataset)
    acc = solver.evaluate(metric='acc')

    print("acc: ", acc)
