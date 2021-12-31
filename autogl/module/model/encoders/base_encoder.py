import torch
import typing as _typing
from ...hpo import AutoModule


class BaseEncoderMaintainer(AutoModule):
    def __init__(
            self,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        super(BaseEncoderMaintainer, self).__init__(
            device, *args, **kwargs
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


class AutoHomogeneousEncoderMaintainer(BaseEncoderMaintainer):
    def _initialize(self) -> _typing.Optional[bool]:
        raise NotImplementedError

    def __init__(
            self,
            input_dimension: _typing.Optional[int] = ...,
            final_dimension: _typing.Optional[int] = ...,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        self._input_dimension: _typing.Optional[int] = input_dimension
        self._final_dimension: _typing.Optional[int] = final_dimension
        super(AutoHomogeneousEncoderMaintainer, self).__init__(
            device, *args, **kwargs
        )
        self.__args: _typing.Tuple[_typing.Any, ...] = args
        self.__kwargs: _typing.Mapping[str, _typing.Any] = kwargs

    @property
    def input_dimension(self) -> _typing.Optional[int]:
        return self._input_dimension
    
    @input_dimension.setter
    def input_dimension(self, input_dimension):
        self._input_dimension = input_dimension

    @property
    def final_dimension(self):
        return self._final_dimension
    
    @final_dimension.setter
    def final_dimension(self, final_dimension):
        # TODO: may mutate search space according to the final dimension
        self._final_dimension = final_dimension

    def from_hyper_parameter(
            self, hyper_parameter: _typing.Mapping[str, _typing.Any], **kwargs
    ):
        new_kwargs = dict(self.__kwargs)
        new_kwargs.update(kwargs)
        duplicate: AutoHomogeneousEncoderMaintainer = self.__class__(
            self.input_dimension, self.final_dimension, self.device,
            **new_kwargs
        )
        hp = dict(self.hyper_parameters)
        hp.update(hyper_parameter)
        duplicate.hyper_parameters = hp
        duplicate.initialize()
        return duplicate

    def get_output_dimensions(self) -> _typing.Iterable[int]:
        """"""
        ''' Note that this is a default implicit assumption '''
        _output_dimensions = [self._input_dimension]
        _output_dimensions.extend(self.hyper_parameters["hidden"])
        if (
                isinstance(self.final_dimension, int) and
                self.final_dimension > 0
        ):
            _output_dimensions.append(self.final_dimension)
        return _output_dimensions
