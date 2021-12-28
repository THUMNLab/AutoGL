import os
os.environ["AUTOGL_BACKEND"] = "pyg"
import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import Acc
from autogl.solver.utils import set_seed
import argparse

if __name__ == '__main__':
    set_seed(202106)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/nodeclf_nas_gasso.yml')
    parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed'], default='citeseer', type=str)

    args = parser.parse_args()

    #dataset = build_dataset_from_name(args.dataset, path = "/DATA/DATANAS1/qinyj/enhgnas/")
    dataset = build_dataset_from_name(args.dataset, path = "~/AGL/")
    solver = AutoNodeClassifier.from_config(args.config)
    solver.fit(dataset)
    solver.get_leaderboard().show()
    out = solver.predict_proba()
    print('acc on dataset', Acc.evaluate(out, dataset[0].y[dataset[0].test_mask].detach().numpy()))
