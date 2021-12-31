import os
import typing as _typing
from autogl.utils import universal_registry
from autogl.data import Dataset


class DatasetUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_dataset(cls, dataset_name: str) -> _typing.Callable[
        [_typing.Type[Dataset]], _typing.Type[Dataset]
    ]:
        def register_dataset_cls(dataset: _typing.Type[Dataset]):
            if not issubclass(dataset, Dataset):
                raise TypeError
            else:
                cls[dataset_name] = dataset
                return dataset

        return register_dataset_cls

    @classmethod
    def get_dataset(cls, dataset_name: str) -> _typing.Type[Dataset]:
        return cls[dataset_name]


def build_dataset_from_name(dataset_name: str, path: str = "~/.cache-autogl/"):
    """

    Parameters
    ----------
    dataset_name: `str`
        name of dataset
    path: `str`
        local cache directory for datasets, default to "~/.cache-autogl/"

    Returns
    -------
    instance of dataset
    """
    path = os.path.expanduser(os.path.join(path, "data", dataset_name))
    _dataset = DatasetUniversalRegistry.get_dataset(dataset_name)
    return _dataset(path)
