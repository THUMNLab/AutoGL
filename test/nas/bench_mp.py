import torch.multiprocessing as mp
from queue import Queue
import time 
import random
import os
def mp_exec(resources,configs,func):
    '''
    @ resources : list of gpu devices
    @ configs : list of params
    @ func : f(dev,cfg)
    '''
    q=Queue()
    for res in resources:
        q.put(res)
    pool=mp.Pool()
    def put_back_dev(dev,cfg):
        def callback(*args):
            print(f"Device {dev} Finish cfg {cfg} ")
            q.put(dev)
            print(*args)
        return callback

    for idx,cfg in enumerate(configs):
        dev = q.get()
        print(f"Start config {cfg} on device {dev}")
        pool.apply_async(func,args=[dev,cfg],callback=put_back_dev(dev,cfg),error_callback=put_back_dev(dev,cfg))

    pool.close()
    pool.join()

# grid_list={
#     "data":'arxiv citeseer computers cora cs photo physics proteins pubmed'.split(),
#     "algo":'graphnas agnn'.split()
# }

grid_list={
    "data":'cora photo arxiv'.split(),
    "algo":'graphnas agnn'.split()
}

param_names=list(grid_list.keys())

resources=[0,1,2,3,4,5,6,7,8]

def gen_config(ni,config,configs):
    if ni>=len(param_names):
        configs.append(config.copy())
        return 
    pname=param_names[ni]
    plist=grid_list[pname]
    for p in plist:
        config.append(p)
        gen_config(ni+1,config,configs)
        config.pop()
def get_configs():
    configs=[]
    config=[]
    gen_config(0,config,configs)
    return configs
def run():
    configs=get_configs()
    mp_exec(resources,configs,func)

import pandas as pd
fdir='./logs/'
def func(dev,cfg):
    cmd=f'CUDA_VISIBLE_DEVICES={dev} AUTOGL_BACKEND=pyg python bench.py --log_dir "{fdir}"'
    for i,pname in enumerate(param_names):
        cmd+=f' --{pname} {cfg[i]}'
    print(cmd)
    os.system(cmd)
def show():
    res=[]
    for r,ds,fs in os.walk(fdir):
        for f in fs:
            if 'log' in f:
                data,algo=eval(os.path.splitext(f)[0])
                with open(os.path.join(r,f)) as file:
                    metric=float(file.read())
                res.append([data,algo,metric])
    df=pd.DataFrame(res,columns='data algo v'.split()).pivot_table(values='v',index='algo',columns='data')
    print(df.to_string())
    df.to_csv(os.path.join(fdir,'results.csv'))

if __name__=='__main__':
    import argparse
    from argparse import ArgumentParser
    def get_args(args=None):
        parser=ArgumentParser()
        parser.add_argument('-t',type=str,default='show',choices=['show','run','debug'])
        args=parser.parse_args(args)
        return args

    args=get_args()
    t=args.t

    if t=='show':
        show()
    elif t=='run':
        run()
    elif t=='debug':
        func(0,get_configs()[0])