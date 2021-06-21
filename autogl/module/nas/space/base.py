from abc import abstractmethod
from autogl.module.model import BaseModel
import torch.nn as nn
from nni.nas.pytorch import mutables

class OrderedMutable():
    def __init__(self, order):
        self.order = order

class OrderedLayerChoice(OrderedMutable, mutables.LayerChoice):
    def __init__(self, order, op_candidates, reduction="sum", return_mask=False, key=None):
        OrderedMutable.__init__(self, order)
        mutables.LayerChoice.__init__(self, op_candidates, reduction, return_mask, key)

class OrderedInputChoice(OrderedMutable, mutables.InputChoice):
    def __init__(self, order, n_candidates=None, choose_from=None, n_chosen=None,
                 reduction="sum", return_mask=False, key=None):
        OrderedMutable.__init__(self, order)
        mutables.InputChoice.__init__(self, n_candidates, choose_from, n_chosen,
                 reduction, return_mask, key)

class BaseSpace(nn.Module):
    """
    Base space class of NAS module. Defining space containing all models.
    Please use mutables to define your whole space. Refer to
    `https://nni.readthedocs.io/en/stable/NAS/WriteSearchSpace.html`
    for detailed information.

    Parameters
    ----------
    init: bool
        Whether to initialize the whole space. Default: `False`
    """

    def __init__(self, init=False):
        super().__init__()
        self._initialized = False

    @abstractmethod
    def _instantiate(self):
        """
        Instantiate modules in the space
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the forward pass of space model
        """
        raise NotImplementedError()

    @abstractmethod
    def export(self, selection: dict, device) -> BaseModel:
        """
        Export the searched model from space.

        Parameters
        ----------
        selection: Dict
            The dictionary containing all the choices of nni.
        device: str or torch.device
            The device to put model on.

        Return
        ------
        model: autogl.module.model.BaseModel
            model to be exported.
        """
        raise NotImplementedError()

    def instantiate(self, *args, **kwargs):
        self._default_key = 0
        self._instantiate(*args, **kwargs)
        if not self._initialized:
            self._initialized = True

    def setLayerChoice(self, order, op_candidates, reduction="sum", return_mask=False, orikey=None):
        """
        Give a unique key if not given
        """
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikry = key
        layer = OrderedLayerChoice(order, op_candidates, reduction, return_mask, orikey)
        return layer

    def setInputChoice(self, order, n_candidates=None, choose_from=None, n_chosen=None,
                 reduction="sum", return_mask=False, orikey=None):
        """
        Give a unique key if not given
        """
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        layer = OrderedInputChoice(order, n_candidates, choose_from, n_chosen,
                 reduction, return_mask, orikey)
        return layer
