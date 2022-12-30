import typing
from autogl.utils import universal_registry
from . import _data_preprocessor


class DataPreprocessorUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_data_preprocessor(cls, name: str) -> typing.Callable[
        [typing.Type[_data_preprocessor.DataPreprocessor]],
        typing.Type[_data_preprocessor.DataPreprocessor]
    ]:
        def register_data_preprocessor(
                data_preprocessor: typing.Type[_data_preprocessor.DataPreprocessor]
        ) -> typing.Type[_data_preprocessor.DataPreprocessor]:
            if not issubclass(data_preprocessor, _data_preprocessor.DataPreprocessor):
                raise TypeError
            else:
                cls[name] = data_preprocessor
                return data_preprocessor

        return register_data_preprocessor

    @classmethod
    def get_data_preprocessor(cls, name: str) -> typing.Type[_data_preprocessor.DataPreprocessor]:
        if name not in cls:
            raise ValueError(f"Data Preprocessor with name \"{name}\" not exist")
        else:
            return cls[name]
