import sys
sys.path.append('../')

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoGraphClassifier
from autogl.module import Acc
import yaml
import random
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
import random
import numpy as np
import torch
import os
import yaml
from autogl.module.feature import FEATURE_DICT
from argparse import ArgumentParser
parser = ArgumentParser()
# parser.add_argument('--device', default=0, type=int)
# parser.add_argument('--max_eval', default=10, type=int)
parser.add_argument('--sn',default=5,type=int)
parser.add_argument('--output',default='./record_gcl2.txt',type=str)
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

def run_gcl(dataset,configs,features,seed):

    print(f'run {dataset} \t {configs} \t {features} \t {seed}')
    setseed(seed)
    dataset = build_dataset_from_name(dataset)
    configs = yaml.load(open(configs, 'r').read(), Loader=yaml.FullLoader)
    configs['features']=[]
    for f in features:
        configs['feature'].append({'name':f})
    
    autoClassifier = AutoGraphClassifier.from_config(configs)
    # train
    autoClassifier.fit(
        dataset, 
        time_limit=3600, 
        train_split=0.8, 
        val_split=0.1, 
        cross_validation=True,
        cv_split=10, 
    )
    # test
    predict_result = autoClassifier.predict_proba()
    acc=Acc.evaluate(predict_result, dataset.data.y[dataset.test_index].cpu().detach().numpy())
    # print(acc)
    return acc

if __name__ == "__main__":
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
            'netlsd',
            'NxGraph', 'NxLargeCliqueSize', 'NxAverageClusteringApproximate', 'NxDegreeAssortativityCoefficient', 'NxDegreePearsonCorrelationCoefficient', 'NxHasBridge', 'NxGraphCliqueNumber', 'NxGraphNumberOfCliques', 'NxTransitivity', 'NxAverageClustering', 'NxIsConnected', 'NxNumberConnectedComponents', 'NxIsDistanceRegular', 'NxLocalEfficiency', 'NxGlobalEfficiency', 'NxIsEulerian'
        ]
    datasets=[
        'mutag',
        'imdb-b',
        'imdb-m',
        'proteins',
        'collab'
    ]
    models=['gin']
    cnt=0
    for fi,f in enumerate(feature_set):
        for mi,m in enumerate(models):
            for di,d in enumerate(datasets):
                for si,seed in enumerate(seeds):
                    cnt+=1
                    if cnt<=0:
                        continue
                    fs=['onlyconst',f] if f !='' else ['onlyconst','graph']
                    try:
                        # queue_configs.append([d,f'../configs/gcl_{m}.yaml',fs,seed])
                        acc=run_gcl(d,f'../configs/gcl_{m}.yaml',fs,seed)   
                    except Exception as e:
                        print(e)
                        acc=-1
                    record_file.write(f'{cnt},{acc},{m},{d},{f},{seed}\n')
                    record_file.flush()
