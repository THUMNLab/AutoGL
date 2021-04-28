from abc import abstractmethod
from autogl.module.model import BaseModel
import torch.nn as nn
from nni.nas.pytorch import mutables

class OrderedMutable():
    def __init__(self, order):
        self.order = order

class OrderedLayerChoice(OrderedMutable, mutables.LayerChoice):
    def __init__(self, order, *args, **kwargs):
        OrderedMutable.__init__(self, order)
        mutables.LayerChoice.__init__(self, *args, **kwargs)

class OrderedInputChoice(OrderedMutable, mutables.InputChoice):
    def __init__(self, order, *args, **kwargs):
        OrderedMutable.__init__(self, order)
        mutables.InputChoice.__init__(self, *args, **kwargs)

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

    def setLayerChoice(self, *args, **kwargs):
        """
        Give a unique key if not given
        """
        if len(args) < 5 and not "key" in kwargs:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            kwargs["key"] = key
        layer = OrderedLayerChoice(*args, **kwargs)
        return layer

    def setInputChoice(self, *args, **kwargs):
        """
        Give a unique key if not given
        """
        if len(args) < 7 and not "key" in kwargs:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            kwargs["key"] = key
        layer = OrderedInputChoice(*args, **kwargs)
        return layer
