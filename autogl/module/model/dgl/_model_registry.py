import typing as _typing
from .base import BaseAutoModel

MODEL_DICT: _typing.Dict[str, _typing.Type[BaseAutoModel]] = {}


def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_DICT:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))
        if not issubclass(cls, BaseAutoModel):
            raise ValueError(
                "Trainer ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_DICT[name] = cls
        return cls

    return register_model_cls


class ModelUniversalRegistry:
    @classmethod
    def get_model(cls, name: str) -> _typing.Type[BaseAutoModel]:
        if type(name) != str:
            raise TypeError(f"Expect model type str, but get {type(name)}.")
        if name not in MODEL_DICT:
            raise KeyError(f"Do not support {name} model in pyg backend")
        return MODEL_DICT.get(name)
