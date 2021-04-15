from copy import deepcopy
import sys
from nni.nas.pytorch.fixed import apply_fixed_architecture
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
sys.path.append('../')
import torch
from autogl.solver import AutoNodeClassifier
from autogl.module.nas.nas import DartsNodeClfEstimator
from autogl.module.nas.space import GraphSpace
from autogl.datasets import build_dataset_from_name
from autogl.module.model import BaseModel
from autogl.module.nas.darts import Darts
from autogl.utils import get_logger

class MyGraphSpace(GraphSpace):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None, init=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ops = ops
        self._initialized = False
        if init:
            self.instantiate()
        
    def instantiate(self, input_dim=None, hidden_dim=None, output_dim=None, ops=None):
        self.input_dim = input_dim or self.input_dim
        self.hidden_dim = hidden_dim or self.hidden_dim
        self.output_dim = output_dim or self.output_dim
        self.ops = ops or self.ops
        super().instantiate(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            ops=self.ops
        )
        self._initialized = True

class SpaceModel(BaseModel):
    def __init__(self, space_model: MyGraphSpace, selection, device=torch.device('cuda')):
        super().__init__(init=True)
        space_model.reinstantiate()
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model.to(device)
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.selection = selection
        apply_fixed_architecture(self._model, selection, verbose=False)
        self.params = {
            "num_class": self.num_classes,
            "features_num": self.num_features
        }
        self.device = device

    def to(self, device):
        if isinstance(device, (str, torch.device)):
            self.device = device
        return super().to(device)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    def from_hyper_parameter(self, hp):
        """
        receive no hp, just copy self and reset the learnable parameters.
        """
        ret_self = deepcopy(self)
        ret_self._model.reinstantiate()
        apply_fixed_architecture(ret_self._model, ret_self.selection, verbose=False)
        ret_self.to(self.device)
        return ret_self

    @property
    def model(self):
        return self._model

class MyDarts(Darts):
    def __init__(self, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def to(self, device):
        """
        change the device of the whole process
        """
        self.device = device

    def search(self, space, dset, trainer):
        """
        TODO: please manage device when training
        current device of search seems to be forced on CPU.
        """
        res = super().search(space, dset, trainer)
        selection = self.export()
        return SpaceModel(res, selection, self.device)

if __name__ == '__main__':
    dataset = build_dataset_from_name('cora')
    solver = AutoNodeClassifier(
        feature_module=None,
        graph_models=[],
        hpo_module="random",
        max_evals=10,
        ensemble_module=None,
        nas_algorithms=[Darts()],
        nas_spaces=[GraphSpace(hidden_dim=64, ops=[GATConv, GCNConv])],
        nas_estimators=[DartsNodeClfEstimator()]
    )
    solver.fit(dataset)
    out = solver.predict(dataset)