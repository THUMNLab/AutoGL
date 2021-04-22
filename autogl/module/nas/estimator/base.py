"""
Base estimator of NAS
"""

from abc import abstractmethod
from ..space import BaseSpace
from typing import Tuple
import torch


class BaseEstimator:
    """
    The estimator of NAS model.
    """

    @abstractmethod
    def infer(
        self, model: BaseSpace, dataset, mask="train"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss and metrics of given model on given dataset using
        specified masks.

        Parameters
        ----------
        model: autogl.module.nas.space.BaseSpace
            The model in space.

        dataset: autogl.dataset
            The dataset to perform infer

        mask: str
            The mask to evalute on dataset

        Return
        ------
        metric: torch.Tensor
            the metric on given datasets.
        loss: torch.Tensor
            the loss on given datasets. Note that loss should be differentiable.
        """
        raise NotImplementedError()
