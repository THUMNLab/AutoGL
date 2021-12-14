import typing as _typing


class _UniversalRegistryMetaclass(type, _typing.MutableMapping[str, _typing.Any]):
    def __getitem__(cls, k: str) -> _typing.Any:
        return cls.__universal_registry[k]

    def __setitem__(cls, k: str, v: _typing.Any) -> None:
        cls.__universal_registry[k] = v

    def __delitem__(cls, k: str) -> None:
        del cls.__universal_registry[k]

    def __len__(cls) -> int:
        return len(cls.__universal_registry)

    def __iter__(cls) -> _typing.Iterator[str]:
        return iter(cls.__universal_registry)

    @property
    def _universal_registry(cls) -> _typing.Mapping[str, _typing.Any]:
        return cls.__universal_registry

    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        return super(_UniversalRegistryMetaclass, mcs).__new__(
            mcs, name, bases, namespace
        )

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_UniversalRegistryMetaclass, cls).__init__(
            name, bases, namespace
        )
        cls.__universal_registry: _typing.MutableMapping[str, _typing.Any] = {}


class UniversalRegistryBase(metaclass=_UniversalRegistryMetaclass):
    ...
