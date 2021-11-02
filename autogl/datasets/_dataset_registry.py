import os
import typing as _typing
from autogl.data import Dataset


class _DatasetUniversalRegistryMetaclass(type):
    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        return super(_DatasetUniversalRegistryMetaclass, mcs).__new__(
            mcs, name, bases, namespace
        )

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_DatasetUniversalRegistryMetaclass, cls).__init__(name, bases, namespace)
        cls._dataset_universal_registry: _typing.MutableMapping[str, _typing.Type[Dataset]] = {}


class DatasetUniversalRegistry(metaclass=_DatasetUniversalRegistryMetaclass):
    @classmethod
    def register_dataset(cls, dataset_name: str):
        def register_dataset_cls(dataset: _typing.Type[Dataset]):
            if dataset_name in cls._dataset_universal_registry:
                raise ValueError(f"Dataset with name \"{dataset_name}\" already exists!")
            elif not issubclass(dataset, Dataset):
                raise TypeError
            else:
                cls._dataset_universal_registry[dataset_name] = dataset
                return dataset

        return register_dataset_cls

    @classmethod
    def get_dataset(cls, dataset_name: str) -> _typing.Type[Dataset]:
        return cls._dataset_universal_registry.get(dataset_name)


def build_dataset_from_name(dataset_name: str, path: str = "~/.cache-autogl/"):
    path = os.path.expanduser(os.path.join(path, "data", dataset_name))
    _dataset = DatasetUniversalRegistry.get_dataset(dataset_name)
    return _dataset(path)
