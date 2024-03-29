import os
os.environ['AUTOGL_BACKEND']='pyg'
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.solver.utils import set_seed
import argparse
from autogl.backend import DependentBackend

if __name__ == '__main__':
    set_seed(202106)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/nodeclf_nas_autoattend_benchmark.yml')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='cora', type=str)

    args = parser.parse_args()

    dataset = build_dataset_from_name(args.dataset)
    if DependentBackend.is_pyg():
        label = dataset[0].y[dataset[0].test_mask].cpu().numpy()
    else:
        label = dataset[0].ndata['label'][dataset[0].ndata['test_mask']]
    # label = dataset[0].nodes.data["y" if DependentBackend.is_pyg() else "label"][dataset[0].nodes.data["test_mask"]].cpu().numpy()
    solver = AutoNodeClassifier.from_config(args.config)
    solver.fit(dataset)
    solver.get_leaderboard().show()
    acc = solver.evaluate(metric="acc")
    print('acc on dataset', acc)
