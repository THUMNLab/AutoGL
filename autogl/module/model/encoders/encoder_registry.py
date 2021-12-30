import typing as _typing
from autogl.utils import universal_registry
from . import base_encoder


class EncoderUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_encoder(cls, name: str) -> _typing.Callable[
        [_typing.Type[base_encoder.BaseEncoderMaintainer]],
        _typing.Type[base_encoder.BaseEncoderMaintainer]
    ]:
        def register_encoder(
                _encoder: _typing.Type[base_encoder.BaseEncoderMaintainer]
        ) -> _typing.Type[base_encoder.BaseEncoderMaintainer]:
            if not issubclass(_encoder, base_encoder.BaseEncoderMaintainer):
                raise TypeError
            else:
                cls[name] = _encoder
                return _encoder

        return register_encoder

    @classmethod
    def get_encoder(cls, name: str) -> _typing.Type[base_encoder.BaseEncoderMaintainer]:
        if name not in cls:
            raise ValueError(f"Encoder with name \"{name}\" not exist")
        else:
            return cls[name]
