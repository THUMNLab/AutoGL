import torch
import typing as _typing
from ...hpo import AutoModule
from ..encoders import base_encoder


class BaseDecoderMaintainer(AutoModule):
    def _initialize(
            self, encoder: base_encoder.AutoHomogeneousEncoderMaintainer, *args, **kwargs
    ) -> _typing.Optional[bool]:
        """ Abstract initialization method to override """
        raise NotImplementedError

    def from_hyper_parameter_and_encoder(
            self, hp: _typing.Mapping[str, _typing.Any],
            encoder: base_encoder.BaseEncoderMaintainer
    ) -> 'BaseDecoderMaintainer':
        duplicate = self.__class__(
            self.output_dimension, self.device
        )
        new_hp = dict(self.hyper_parameters)
        new_hp.update(hp)
        duplicate.hyper_parameters = new_hp
        duplicate.initialize(encoder)
        return duplicate

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(BaseDecoderMaintainer, self).__init__(
            device, *args, **kwargs
        )
        self.output_dimension = output_dimension
        self._decoder: _typing.Optional[torch.nn.Module] = None

    @property
    def output_dimension(self):
        return self.__output_dimension
    
    @output_dimension.setter
    def output_dimension(self, output_dimension):
        self.__output_dimension = output_dimension

    @property
    def decoder(self) -> _typing.Optional[torch.nn.Module]:
        return self._decoder

    def to_device(self, device: _typing.Union[torch.device, str, int, None]):
        self.device = device
        if (
                self._decoder not in (Ellipsis, None) and
                isinstance(self._decoder, torch.nn.Module)
        ):
            self._decoder.to(self.device)
