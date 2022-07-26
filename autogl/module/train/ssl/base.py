import os
import torch
import torch_geometric
import logging

import numpy as np
import typing as _typing

import torch.nn.functional
import torch.utils.data

from dig.sslgraph.method.contrastive.objectives import NCE_loss, JSE_loss

from autogl.module.model.encoders.base_encoder import AutoHomogeneousEncoderMaintainer

from ..base import BaseTrainer
from ..evaluation import Evaluation, get_feval, Acc
from ...model import (
    BaseAutoModel,
    BaseEncoderMaintainer, 
    BaseDecoderMaintainer,
    EncoderUniversalRegistry,
    DecoderUniversalRegistry
)

class BaseContrastiveTrainer(BaseTrainer):
    def __init__(
        self, 
        encoder: _typing.Union[BaseEncoderMaintainer, str, None], 
        decoder: _typing.Union[BaseDecoderMaintainer, str, None],
        decoder_node: _typing.Union[BaseDecoderMaintainer, None],
        num_features: _typing.Union[int, None] = None,
        num_graph_features: _typing.Union[int, None] = None,
        device: _typing.Union[torch.device, str] = "auto",
        feval: _typing.Union[
            _typing.Sequence[str], _typing.Sequence[_typing.Type[Evaluation]]
        ] = (Acc,),
        loss: _typing.Union[str, _typing.Callable] = "NCE",
        views_fn: _typing.Union[
            _typing.Sequence[_typing.Callable], None
        ] = None,
        graph_level: bool = True,
        node_level: bool = False,
        z_dim: _typing.Union[int, None] = None,
        z_node_dim: _typing.Union[int, None] = None,
        neg_by_crpt: bool = False,
        tau: int = 0.5,
        model_path: _typing.Union[str, None] = "./models"
    ):
        """
        The basic trainer for self-supervised learning with contrastive method.
        Used to automatically train the self-supervised problems.
        Parameters
        ----------
        encoder: `BaseEncoderMaintainer`, `str` or None
            A graph encoder shared by all views.
        decoder: `BaseDecoderMaintainer`, `str` or None
            A decoder which can be understood as a projection head for graph-level representations.
            Only required if `graph_level` = True.
        decoder_node: `BaseDecoderMaintainer`, `str` or None
            A decoder which can be understood as a projection head for node-level representations.
            Only required if `node_level` = True.
        num_features: `int` or None, Optional
            The number of features in dataset.
        num_graph_features: `int` or None, Optional
            The number of graph level features in dataset.
        device: `torch.device` or `str`, Optional
            The device this trainer will use.
            When `device` = "auto", if GPU exists in the device and dependency is installed, 
            the trainer will give priority to GPU, otherwise CPU will be used
        feval: a sequence of `str` or a sequence of `Evaluation`, Optional
            The evaluation methods.
        loss: `str` or `Callable`, Optional
            The loss function or the learning objective of contrastive model.
        views_fn: a list of `Callable` or None, Optional
            List of functions or augmentation methods to generate views from give graphs.
        graph_level: `bool`, Optional
            Whether to include graph-level representations
        node_level: `bool`, Optional
            Whether to include node-level representations
        z_dim: `int`, Optional
            The dimension of graph-level representations
        z_node_dim: `int`, Optional
            The dimension of node-level representations
        neg_by_crpt: `bool`, Optional
            The mode to obtain negative samples
        tau: `int`, Optional
            The temperature parameter in InfoNCE loss. Only used when `loss` = "NCE"
        model_path: `str` or None, Optional
            The directory to restore the saved model.
            If `model_path` = None, the model will not be saved.
        """
        assert (node_level or graph_level) is True
        assert not (loss == "NCE" and neg_by_crpt)
        assert isinstance(encoder, BaseEncoderMaintainer) or isinstance(encoder, str)
        self.loss = self._get_loss(loss)
        self.views_fn = views_fn
        self.node_level = node_level
        self.graph_level = graph_level
        self.z_dim = z_dim
        self.z_node_dim = z_node_dim
        self._encoder = None
        # TODO
        # do not support method with both node-level representation and graph-level representation
        # so the decoder will be either _decoder or _decoder_node, one of them
        self._decoder = None
        self.last_dim = z_dim if graph_level else z_node_dim
        self.num_features = num_features
        self.num_graph_features = num_graph_features
        self.neg_by_crpt = neg_by_crpt
        self.tau = tau
        self.model_path = model_path
        if isinstance(device, str):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            encoder=encoder,
            decoder=decoder if graph_level else decoder_node,
            device=self.device,
            feval=feval,
            loss=self.loss
        )

    def _get_loss(self, loss):
        if callable(loss):
            return loss
        elif isinstance(loss, str):
            assert loss in ['JSE', 'NCE']
            return {'JSE': JSE_loss, 'NCE': NCE_loss}[loss]
        else:
            raise NotImplementedError("The argument `loss` should be str or callable which returns a loss tensor")

    # override encoder and decoder to depend on contrastive learning
    @property
    def encoder(self):
        return self._encoder
    
    @encoder.setter
    def encoder(self, enc: _typing.Union[BaseEncoderMaintainer, str, None]):
        if isinstance(enc, str):
            if enc in EncoderUniversalRegistry:
                if self.node_level:
                    self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                        self.num_features, final_dimension=self.last_dim, device=self.device, init=self.initialized
                    )
                elif self.graph_level:
                    self._encoder = EncoderUniversalRegistry.get_encoder(enc)(
                        self.num_features,
                        final_dimension=self.last_dim,
                        num_graph_featues=self.num_graph_features,
                        device=self.device,
                        init=self.initialized
                    )
            else:
                raise NotImplementedError(f"Sorry. Encoder {enc} is not supported yet.")
        elif isinstance(enc, BaseEncoderMaintainer):
            self._encoder = enc
        elif enc is None:
            self._encoder = None
        else:
            raise NotImplementedError(f"Sorry. Encoder {enc} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features
    
    @property
    def decoder(self):
        if isinstance(self.encoder, BaseAutoModel): return None
        return self._decoder

    @decoder.setter
    def decoder(self, dec: _typing.Union[BaseDecoderMaintainer, str, None]):
        if isinstance(self.encoder, BaseAutoModel):
            logging.warn("Ignore passed dec since enc is a whole model")
            self._decoder = None
            return
        if isinstance(dec, str):
            if self.node_level:
                self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                    self.last_dim,
                    input_dimension=self.last_dim,
                    device=self.device,
                    init=self.initialized
                )
            elif self.graph_level:
                self._decoder = DecoderUniversalRegistry.get_decoder(dec)(
                    self.last_dim,
                    input_dimension=self.last_dim,
                    num_graph_features=self.num_graph_features,
                    device=self.device,
                    init=self.initialized
                )
        elif isinstance(dec, BaseDecoderMaintainer) or dec is None:
            self._decoder = dec
        else:
            raise NotImplementedError(f"Sorry. The decoder {dec} is not supported yet.")
        self.num_features = self.num_features
        self.last_dim = self.last_dim
        self.num_graph_features = self.num_graph_features

    @property
    def num_graph_features(self):
        return self._num_graph_features

    @num_graph_features.setter
    def num_graph_features(self, num_graph_featues):
        self._num_graph_features = num_graph_featues
        if self.graph_level:
            if self.encoder is not None: self.encoder.num_graph_features = self._num_graph_features
            if self.decoder is not None: self.decoder.num_graph_features = self._num_graph_features

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, num_features):
        self._num_features = num_features
        if self.encoder is not None:
            self.encoder.input_dimension = num_features
