import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNAS
from ..space import BaseSpace
from ..utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, get_module_order, sort_replaced_module
from nni.nas.pytorch.fixed import apply_fixed_architecture
from tqdm import tqdm
_logger = logging.getLogger(__name__)
from .rl import PathSamplingLayerChoice,PathSamplingInputChoice
import numpy as np
class RSBox:
    '''get selection space for model `space` '''
    def __init__(self,space):
        self.model = space
        self.nas_modules = []
        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules) 
        nm=self.nas_modules
        selection_range={}
        for k,v in nm:
            selection_range[k]=len(v)
        self.selection_dict=selection_range
        
        
        space_size=np.prod(list(selection_range.values()))
        print(f'Using random search Box. Total space size: {space_size}')
        print('Searching Space:',selection_range)
    def export(self):
        return self.selection_dict #{k:v}, means action ranges 0 to v-1 for layer named k
    def sample(self):
        # uniformly sample
        selection={}
        sdict=self.export()
        for k,v in sdict.items():
            selection[k]=np.random.choice(range(v))
        return selection

class RandomSearch(BaseNAS):
    '''
    uniformly search
    '''
    def __init__(self, device='cuda',num_epochs=400,disable_progress=False,*args,**kwargs):
        super().__init__(device)
        self.num_epochs=num_epochs
        self.disable_progress=disable_progress
    def search(self, space: BaseSpace, dset, estimator):
        self.estimator=estimator
        self.dataset=dset
        self.space=space
        self.box=RSBox(self.space)
        arch_perfs=[]
        cache={}
        with tqdm(range(self.num_epochs),disable=self.disable_progress) as bar:
            for i in bar:
                selection=self.export() 
                # print(selection)
                vec=tuple(list(selection.values()))
                if vec not in cache:
                    self.arch=space.export(selection,self.device)
                    metric,loss=self._infer(mask='val')
                    arch_perfs.append([metric,selection])
                    cache[vec]=metric
                bar.set_postfix(acc=metric,max_acc=max(cache.values()))
        selection=arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        arch=space.export(selection,self.device)
        return arch 
    
    def export(self):
        arch=self.box.sample()
        return arch

    def _infer(self,mask='train'):
        metric, loss = self.estimator.infer(self.arch, self.dataset,mask=mask)
        return metric, loss
