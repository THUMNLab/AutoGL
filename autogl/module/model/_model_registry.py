import typing as _typing
from .base import BaseModel

MODEL_DICT: _typing.Dict[str, _typing.Type[BaseModel]] = {}


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_DICT:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_DICT[name] = cls
        return cls

    return register_model_cls


class ModelUniversalRegistry:
    @classmethod
    def get_model(cls, name: str) -> _typing.Type[BaseModel]:
        if type(name) != str:
            raise TypeError
        if name not in MODEL_DICT:
            raise KeyError
        return MODEL_DICT.get(name)
