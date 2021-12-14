import typing as _typing
from autogl.utils import universal_registry
from . import base_encoder


class EncoderUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_encoder(cls, name: str) -> _typing.Callable[
        [_typing.Type[base_encoder.BaseAutoEncoderMaintainer]],
        _typing.Type[base_encoder.BaseAutoEncoderMaintainer]
    ]:
        def register_encoder(
                _encoder: _typing.Type[base_encoder.BaseAutoEncoderMaintainer]
        ) -> _typing.Type[base_encoder.BaseAutoEncoderMaintainer]:
            if not issubclass(_encoder, base_encoder.BaseAutoEncoderMaintainer):
                raise TypeError
            elif name in cls:
                raise ValueError
            else:
                cls[name] = _encoder
                return _encoder

        return register_encoder

    @classmethod
    def get_encoder(cls, name: str) -> _typing.Type[base_encoder.BaseAutoEncoderMaintainer]:
        if name not in cls:
            raise ValueError(f"Encoder with name \"{name}\" not exist")
        else:
            return cls[name]
