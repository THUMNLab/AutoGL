"""
Base class for algorithm
"""
from ...model import BaseModel
import torch
from abc import abstractmethod
from ....utils import get_device


class BaseNAS:
    """
    Base NAS algorithm class

    Parameters
    ----------
    device: str or torch.device
        The device of the whole process
    """

    def __init__(self, device="auto") -> None:
        self.device = get_device(device)

    def to(self, device):
        """
        Change the device of the whole NAS search process

        Parameters
        ----------
        device: str or torch.device
        """
        self.device = get_device(device)

    @abstractmethod
    def search(self, space, dataset, estimator) -> BaseModel:
        """
        The search process of NAS.

        Parameters
        ----------
        space : autogl.module.nas.space.BaseSpace
            The search space. Constructed following nni.
        dataset : autogl.datasets
            Dataset to perform search on.
        estimator : autogl.module.nas.estimator.BaseEstimator
            The estimator to compute loss & metrics.

        Returns
        -------
        model: autogl.module.model.BaseModel
            The searched model.
        """
        raise NotImplementedError()
