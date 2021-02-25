import sys

from networkx.algorithms.reciprocity import reciprocity
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
import sys
from numpy.core.defchararray import index
from torch.utils.data import dataset

from yaml import compose, load
sys.path.append('../')
import random
import numpy as np
import torch
import os
import yaml
import re
from autogl.module.feature.base import BaseFeatureAtom
from autogl.module.feature import FEATURE_DICT
import pandas as pd
import copy
from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument('--device', default=0, type=int)
# parser.add_argument('--max_eval', default=10, type=int)
parser.add_argument('--sn',default=5,type=int)
parser.add_argument('--output',default='./record.txt',type=str)
parser.add_argument('--clean',default=False,type=bool)
args=dict()
def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def renew(record_file):
    with open(record_file,'w') as file:
        file.write('')

def run_ncl(dataset,configs,features,seed):
    print(f'run {dataset} \t {configs} \t {features} \t {seed}')
    setseed(seed)
    dataset = build_dataset_from_name(dataset)
    configs = yaml.load(open(configs, 'r').read(), Loader=yaml.FullLoader)
    configs['features']=[]
    for f in features:
        configs['feature'].append({'name':f})
    
    autoClassifier = AutoNodeClassifier.from_config(configs)
    # train
    if dataset in ['cora', 'citeseer', 'pubmed']:
        autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc])
    else:
        autoClassifier.fit(dataset, time_limit=3600, evaluation_method=[Acc], seed=seed, train_split=20*dataset.num_classes, val_split=30*dataset.num_classes, balanced=False)
    val = autoClassifier.get_model_by_performance(0)[0].get_valid_score()[0]
    # print('val acc: ', val)

    # test
    predict_result = autoClassifier.predict_proba(use_best=True, use_ensemble=False)
    test_result=Acc.evaluate(predict_result, dataset.data.y[dataset.data.test_mask].numpy())
    # print('test acc: ', test_result)
    return test_result

if __name__ == '__main__':
    print(f"all FEs {FEATURE_DICT.keys()}")
    args = parser.parse_args()
    
    record_file=args.output
    if not os.path.exists(record_file):
        renew(record_file)
    print(f"record file {record_file}")
    record_file=open(record_file,'a+')

    sn=args.sn # seeds num for each config
    setseed(2021)
    seeds=[random.randint(0,12345678) for x in range(sn)]
    print('setting seeds ',seeds)

    feature_set=[
            '',
            'onehot',
            'PYGOneHotDegree',
            'eigen',
            'pagerank',
            'PYGLocalDegreeProfile',
            'graphlet',
        ]
    datasets=[
        'cora',
        'citeseer',
        'pubmed',
        'amazon_computers',
        'amazon_photo',
        'coauthor_cs',
        'coauthor_physics',
        # 'reddit'
    ]
    models=['gcn','gat']
    cnt=0
    for fi,f in enumerate(feature_set):
        for mi,m in enumerate(models):
            for di,d in enumerate(datasets):
                for si,seed in enumerate(seeds):
                    cnt+=1
                    if cnt<=100:
                        continue
                    fs=['onlyconst',f] if f !='' else ['onlyconst']
                    try:
                        acc=run_ncl(d,f'../configs/ncl_{m}.yaml',fs,seed)   
                    except Exception as e:
                        print(e)
                        acc=-1
                    record_file.write(f'{cnt},{acc},{m},{d},{f},{seed}\n')
                    record_file.flush()


                



    




