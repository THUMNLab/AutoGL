import os
os.environ["AUTOGL_BACKEND"] = 'dgl'

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoHeteroNodeClassifier

if __name__ == '__main__':
    acm = build_dataset_from_name("hetero-acm-han")
    solver = AutoHeteroNodeClassifier(max_evals=10)
    solver.fit(acm)
    acc = solver.evaluate(metric='acc')

    print("acc: ", acc)
