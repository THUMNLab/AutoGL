import sys
sys.path.append('../')
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module import Acc
import yaml
import random
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='cora', type=str)
    parser.add_argument('--configs', type=str, default='../configs/nodeclf_gat_benchmark_small.yml')
    # following arguments will override parameters in the config file
    parser.add_argument('--hpo', type=str, default='random')
    parser.add_argument('--max_eval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    seed = args.seed
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = build_dataset_from_name(args.dataset)
    
    configs = yaml.load(open(args.configs, 'r').read(), Loader=yaml.FullLoader)
    configs['hpo']['name'] = args.hpo
    configs['hpo']['max_evals'] = args.max_eval
    autoClassifier = AutoNodeClassifier.from_config(configs)

    # train
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc])
    else:
        autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc], seed=seed, train_split=20*dataset.num_classes, val_split=30*dataset.num_classes, balanced=False)
    val = autoClassifier.get_model_by_performance(0)[0].get_valid_score()[0]
    print('val acc: ', val)

    # test
    predict_result = autoClassifier.predict_proba(use_best=True, use_ensemble=False)
    print('test acc: ', Acc.evaluate(predict_result, dataset.data.y[dataset.data.test_mask].numpy()))



