import torch
import typing as _typing


class AutoModule:
    def _initialize(self) -> _typing.Optional[bool]:
        """ Abstract initialization method to override """
        raise NotImplementedError

    @property
    def initialized(self) -> bool:
        return self.__initialized

    def initialize(self) -> bool:
        if self.__initialized:
            return self.__initialized
        else:
            init_flag = self._initialize()
            self.__initialized = (
                init_flag if isinstance(init_flag, bool) else True
            )
            return self.__initialized

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, device: _typing.Union[torch.device, str, int, None]):
        if (
                isinstance(device, str) or isinstance(device, int) or
                isinstance(device, torch.device)
        ):
            self.__device = torch.device(device)
        else:
            self.__device = torch.device("cpu")

    def __init__(
            self, initialize: bool,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        self.__hyper_parameter: _typing.Mapping[str, _typing.Any] = {}
        self.__hyper_parameter_space: _typing.Iterable[_typing.Mapping[str, _typing.Any]] = []
        if (
                isinstance(device, str) or isinstance(device, int) or
                isinstance(device, torch.device)
        ):
            self.__device: torch.device = torch.device(device)
        else:
            self.__device: torch.device = torch.device("cpu")
        self.__args: _typing.Tuple[_typing.Any, ...] = args
        self.__kwargs: _typing.Mapping[str, _typing.Any] = kwargs
        self.__initialized: bool = False
        if initialize:
            self.initialize()

    @property
    def hyper_parameter(self) -> _typing.Mapping[str, _typing.Any]:
        return self.__hyper_parameter

    @hyper_parameter.setter
    def hyper_parameter(self, hp: _typing.Mapping[str, _typing.Any]):
        self.__hyper_parameter = hp

    @property
    def hyper_parameters(self) -> _typing.Mapping[str, _typing.Any]:
        return self.__hyper_parameter

    @hyper_parameters.setter
    def hyper_parameters(self, hp: _typing.Mapping[str, _typing.Any]):
        self.__hyper_parameter = hp

    @property
    def hyper_parameter_space(self) -> _typing.Iterable[
        _typing.Mapping[str, _typing.Any]
    ]:
        return self.__hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(
            self, hp_space: _typing.Iterable[_typing.Mapping[str, _typing.Any]]
    ):
        self.__hyper_parameter_space = hp_space
