from abc import abstractmethod
from autogl.module.model import BaseModel
import torch.nn as nn


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
    def instantiate(self):
        """
        Instantiate modules in the space
        """
        if not self._initialized:
            self._initialized = True

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
