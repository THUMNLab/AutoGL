import torch
import typing as _typing
from .._utils import auto_module


class BaseAutoEncoderMaintainer(auto_module.AutoModule):
    def __init__(
            self, initialize: bool,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(BaseAutoEncoderMaintainer, self).__init__(
            initialize, device, *args, **kwargs
        )
        self._encoder: _typing.Optional[torch.nn.Module] = None

    @property
    def encoder(self) -> _typing.Optional[torch.nn.Module]:
        return self._encoder

    def to_device(self, device: _typing.Union[torch.device, str, int, None]):
        self.device = device
        if (
                self._encoder not in (Ellipsis, None) and
                isinstance(self._encoder, torch.nn.Module)
        ):
            self._encoder.to(self.device)

    def from_hyper_parameter(
            self, hyper_parameter: _typing.Mapping[str, _typing.Any], **kwargs
    ):
        raise NotImplementedError

    def _initialize(self) -> _typing.Optional[bool]:
        raise NotImplementedError


class AutoHomogeneousEncoderMaintainer(BaseAutoEncoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        raise NotImplementedError

    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            initialize: bool = False,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        self._input_dimension: _typing.Optional[int] = input_dimension
        self._final_dimension: _typing.Optional[int] = final_dimension
        super(AutoHomogeneousEncoderMaintainer, self).__init__(
            initialize, device, *args, **kwargs
        )
        self.__args: _typing.Tuple[_typing.Any, ...] = args
        self.__kwargs: _typing.Mapping[str, _typing.Any] = kwargs

    def from_hyper_parameter(
            self, hyper_parameter: _typing.Mapping[str, _typing.Any], **kwargs
    ):
        new_kwargs = dict(self.__kwargs)
        new_kwargs.update(kwargs)
        duplicate: AutoHomogeneousEncoderMaintainer = self.__class__(
            self._input_dimension, self._final_dimension,
            False, self.device, **new_kwargs
        )
        hp = dict(self.hyper_parameters)
        hp.update(hyper_parameter)
        duplicate.hyper_parameters = hp
        duplicate.initialize()
        return duplicate

    def get_output_dimensions(self) -> _typing.Iterable[int]:
        """"""
        ''' Note that this is a default implicit assumption '''
        _output_dimensions = list(self.hyper_parameters["hidden"])
        if (
                isinstance(self._final_dimension, int) and
                self._final_dimension > 0
        ):
            _output_dimensions.append(self._final_dimension)
        return _output_dimensions
