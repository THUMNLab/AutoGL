import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_nas_algo
from .base import BaseNAS
from ..space import BaseSpace
from ..utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, get_module_order, sort_replaced_module
from nni.nas.pytorch.fixed import apply_fixed_architecture
from tqdm import tqdm
from .rl import PathSamplingLayerChoice,PathSamplingInputChoice
import numpy as np
from ....utils import get_logger

LOGGER = get_logger("random_search_NAS")
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
        #print(f'Using random search Box. Total space size: {space_size}')
        #print('Searching Space:',selection_range)
         
    def export(self):
        return self.selection_dict #{k:v}, means action ranges 0 to v-1 for layer named k

    def sample(self):
        # uniformly sample
        selection={}
        sdict=self.export()
        for k,v in sdict.items():
            selection[k]=np.random.choice(range(v))
        return selection

@register_nas_algo("random")
class RandomSearch(BaseNAS):
    '''
    Uniformly random architecture search

    Parameters
    ----------
    device : str or torch.device
        The device of the whole process, e.g. "cuda", torch.device("cpu")
    num_epochs : int
        Number of epochs planned for training.
    disable_progeress: boolean
        Control whether show the progress bar.
    '''
    def __init__(self, device='cuda', num_epochs=400, disable_progress=False):
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
                vec=tuple(list(selection.values()))
                if vec not in cache:
                    self.arch=space.parse_model(selection,self.device)
                    metric,loss=self._infer(mask='val')
                    arch_perfs.append([metric,selection])
                    cache[vec]=metric
                bar.set_postfix(acc=metric,max_acc=max(cache.values()))
        selection=arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        arch=space.parse_model(selection,self.device)
        return arch 
    
    def export(self):
        arch=self.box.sample()
        return arch

    def _infer(self,mask='train'):
        metric, loss = self.estimator.infer(self.arch, self.dataset,mask=mask)
        return metric, loss
