import sys
sys.path.append('../')
from torch_geometric.nn import GCNConv
import torch
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import NodeClassificationFullTrainer
from autogl.module.nas import Darts, OneShotEstimator
from autogl.module.nas.space.graph_nas import GraphNasNodeClassificationSpace
from autogl.module.nas.space.graph_nas_macro import GraphNasMacroNodeClassificationSpace
from autogl.module.train import Acc
from autogl.module.nas.algorithm.enas import Enas
from autogl.module.nas.algorithm.rl import RL,GraphNasRL
from autogl.module.nas.estimator.train_scratch import TrainEstimator
from autogl.module.nas.algorithm.random_search import RandomSearch
import numpy as np
import logging
def one_run():
    logging.getLogger().setLevel(logging.WARNING)
    cora = build_dataset_from_name('cora')

    clf = AutoNodeClassifier(
        feature_module='PYGNormalizeFeatures',
        graph_models=[],
        nas_algorithms=[Enas(num_epochs=10)],
        nas_spaces=[GraphNasNodeClassificationSpace()],
        nas_estimators=[OneShotEstimator()],
        max_evals=2
    )

    clf.fit(cora)
    clf.predict(cora)

    return

    dataset = build_dataset_from_name('cora')
    solver = AutoNodeClassifier(
        feature_module='PYGNormalizeFeatures',
        graph_models=[],
        hpo_module=None,
        ensemble_module=None,
        default_trainer=NodeClassificationFullTrainer(
            optimizer=torch.optim.Adam,
            lr=0.005,
            max_epoch=300,
            early_stopping_round=20,
            weight_decay=5e-4,
            device="auto",
            init=False,
            feval=['acc'],
            loss="nll_loss",
            lr_scheduler_type=None,),
        # nas_algorithms=[RL(num_epochs=400)],
        nas_algorithms=[GraphNasRL(num_epochs=1)],
        #nas_algorithms=[Darts(num_epochs=200)],
        nas_spaces=[GraphNasMacroNodeClfSpace(hidden_dim=16,search_act_con=True,layer_number=2)],
        nas_estimators=[TrainEstimator()]
    )

    solver.fit(dataset)
    solver.get_leaderboard().show()
    out = solver.predict_proba()
    acc = Acc.evaluate(out, dataset[0].y[dataset[0].test_mask].detach().numpy())
    print('acc on cora', acc)
    return acc

if __name__ == '__main__':
    acc_li = []
    for i in range(2):
        acc_li.append(one_run())
    print("results:", np.mean(acc_li), np.std(acc_li))