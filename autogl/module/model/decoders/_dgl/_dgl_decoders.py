import torch.nn.functional
import typing as _typing
import dgl
from .. import base_decoder
from ...encoders import base_encoder


class _LogSoftmaxDecoder(torch.nn.Module):
    def forward(
            self, graph: dgl.DGLGraph,
            features: _typing.Sequence[torch.Tensor],
            *args, **kwargs
    ):
        return torch.nn.functional.log_softmax(features[-1], dim=-1)


class LogSoftmaxDecoderMaintainer(base_decoder.BaseAutoDecoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        self._decoder = _LogSoftmaxDecoder().to(self.device)
        return True

    def from_hyper_parameter_and_encoder(
            self, hp: _typing.Mapping[str, _typing.Any],
            encoder: base_encoder.BaseAutoEncoderMaintainer
    ) -> base_decoder.BaseAutoDecoderMaintainer:
        new_hp = dict(self.hyper_parameter)
        new_hp.update(hp)
        duplicate = self.__class__(
            self._output_dimension, False, self.device
        )
        duplicate.hyper_parameter = new_hp
        return duplicate
