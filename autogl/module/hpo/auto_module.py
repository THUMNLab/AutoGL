import torch
import typing as _typing


class AutoModule:
    def _initialize(self, *args, **kwargs) -> _typing.Optional[bool]:
        """ Abstract initialization method to override """
        raise NotImplementedError

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, *args, **kwargs) -> bool:
        if self._initialized:
            return self._initialized
        else:
            init_flag = self._initialize(*args, **kwargs)
            self._initialized = (
                init_flag if isinstance(init_flag, bool) else True
            )
            return self._initialized

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, __device: _typing.Union[torch.device, str, int, None]):
        if type(__device) == torch.device or (
            type(__device) == str and __device.lower() != "auto"
        ) or type(__device) == int:
            self._device: torch.device = torch.device(__device)
        else:
            self._device: torch.device = torch.device(
                "cuda"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )

    def __init__(
            self,
            device: _typing.Union[torch.device, str, int, None] = ...,
            *args, **kwargs
    ):
        self.__hyper_parameters: _typing.Mapping[str, _typing.Any] = dict()
        self.__hyper_parameter_space: _typing.Iterable[_typing.Mapping[str, _typing.Any]] = []
        self.device = device
        self.__args: _typing.Tuple[_typing.Any, ...] = args
        self.__kwargs: _typing.Mapping[str, _typing.Any] = kwargs
        self._initialized: bool = False

    @property
    def hyper_parameters(self) -> _typing.Mapping[str, _typing.Any]:
        return self.__hyper_parameters

    @hyper_parameters.setter
    def hyper_parameters(self, hp: _typing.Mapping[str, _typing.Any]):
        self.__hyper_parameters = hp

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
