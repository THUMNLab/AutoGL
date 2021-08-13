import os
import logging as _logging
import typing as _typing

_backend_logger: _logging.Logger = _logging.getLogger("backend")


class _BackendConfig(_typing.Mapping[str, _typing.Any]):
    def __init__(self, name: str, configurations: _typing.Mapping[str, _typing.Any] = ...):
        self.__name: str = name
        if (
                configurations not in (None, Ellipsis, ...) and
                isinstance(configurations, _typing.Mapping)
        ):
            self.__configurations: _typing.Mapping[str, _typing.Any] = configurations
        else:
            self.__configurations: _typing.Mapping[str, _typing.Any] = dict()

    def __str__(self) -> str:
        return self.__name

    def __getitem__(self, key: str):
        return self.__configurations[key]

    def __len__(self) -> int:
        return len(self.__configurations)

    def __iter__(self):
        return iter(self.__configurations)


class _DGLConfig(_BackendConfig):
    def __init__(self):
        super(_DGLConfig, self).__init__("dgl")


class _PyGConfig(_BackendConfig):
    def __init__(self):
        super(_PyGConfig, self).__init__("pyg")


# class _BackendConfigGenerator:
#     ...


def _generate_backend_config() -> _BackendConfig:
    def _generate_by_name(name: _typing.Optional[str] = ...) -> _BackendConfig:
        if name in (None, Ellipsis, ...) or not isinstance(name, str):
            try:
                import dgl
                return _DGLConfig()
            except ModuleNotFoundError:
                pass
            try:
                import torch_geometric
                return _PyGConfig()
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Neither DGL nor PyTorch-Geometric exists")
        elif name.lower() not in ("dgl", "pyg"):
            __warning_message = " ".join((
                "The environment variable AUTOGL_BACKEND specified",
                "but is neither \"dgl\" nor \"pyg\",",
                "thus the environment variable is ignored",
                "and dependent backend for AutoGL is set automatically",
            ))
            _backend_logger.warning(__warning_message)
            return _generate_by_name()
        elif name.lower() == "dgl":
            try:
                import dgl
                return _DGLConfig()
            except ModuleNotFoundError:
                pass
            try:
                import torch_geometric
                __warning_message: str = " ".join((
                    "The required backend DGL is not installed,",
                    "use PyTorch-Geometric instead.",
                ))
                _backend_logger.warning(__warning_message)
                return _PyGConfig()
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Neither DGL nor PyTorch-Geometric exists")
        elif name.lower() == "pyg":
            try:
                import torch_geometric
                return _PyGConfig()
            except ModuleNotFoundError:
                pass
            try:
                import dgl
                __warning_message: str = " ".join((
                    "The required backend PyTorch-Geometric is not installed,",
                    "use DGL instead.",
                ))
                _backend_logger.warning(__warning_message)
                return _DGLConfig()
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Neither DGL nor PyTorch-Geometric exists")
        else:
            return _generate_by_name()

    if "AUTOGL_BACKEND" in os.environ:
        return _generate_by_name(os.getenv("AUTOGL_BACKEND"))
    else:
        return _generate_by_name()


class _DependentBackendMetaclass(type):
    """
    Metaclass for ``DependentBackend``.
    To ensure the backend config is unique in diverse threads for multiprocessing runtime,
    the backend config is instantiated in the metaclass during interpretation phase.
    """
    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        for base in bases:
            if isinstance(base, _DependentBackendMetaclass):
                strings = (
                    f"{base} is instance of Metaclass {_DependentBackendMetaclass}",
                    f"and MUST not be inherited/extended by <{name}> to construct"
                )
                raise TypeError(" ".join(strings))
        instance = super(_DependentBackendMetaclass, mcs).__new__(mcs, name, bases, namespace)
        return instance

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_DependentBackendMetaclass, cls).__init__(name, bases, namespace)
        cls._backend_config: _BackendConfig = _generate_backend_config()
        _backend_logger.info("Adopted backend: %s" % str(cls._backend_config))


class DependentBackend(metaclass=_DependentBackendMetaclass):
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"The class {DependentBackend} should not be instantiated")

    @classmethod
    def get_backend_name(cls) -> str:
        return str(cls._backend_config)

    @classmethod
    def is_dgl(cls) -> bool:
        return isinstance(cls._backend_config, _DGLConfig)

    @classmethod
    def is_pyg(cls) -> bool:
        return isinstance(cls._backend_config, _PyGConfig)
