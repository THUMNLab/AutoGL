from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoHeteroNodeClassifier

if __name__ == '__main__':
    acm = build_dataset_from_name("hetero-acm-han")
    solver = AutoHeteroNodeClassifier(max_evals=10)
    solver.fit(acm)
    res = solver.predict_proba()
