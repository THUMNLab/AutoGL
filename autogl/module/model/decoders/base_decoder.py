import torch
import typing as _typing
from .._utils import auto_module
from ..encoders import base_encoder


class BaseAutoDecoderMaintainer(auto_module.AutoModule):
    def _initialize(self) -> _typing.Optional[bool]:
        """ Abstract initialization method to override """
        raise NotImplementedError

    def from_hyper_parameter_and_encoder(
            self, hp: _typing.Mapping[str, _typing.Any],
            encoder: base_encoder.BaseAutoEncoderMaintainer
    ) -> 'BaseAutoDecoderMaintainer':
        raise NotImplementedError

    def __init__(
            self, output_dimension: _typing.Optional[int] = ...,
            initialize: bool = False,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(BaseAutoDecoderMaintainer, self).__init__(
            initialize, device, *args, **kwargs
        )
        self._output_dimension: _typing.Optional[int] = output_dimension
        self._decoder: _typing.Optional[torch.nn.Module] = None

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
