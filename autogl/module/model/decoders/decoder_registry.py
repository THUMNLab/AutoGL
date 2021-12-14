import typing as _typing
from autogl.utils import universal_registry
from . import base_decoder


class DecoderUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_decoder(cls, name: str) -> _typing.Callable[
        [_typing.Type[base_decoder.BaseAutoDecoderMaintainer]],
        _typing.Type[base_decoder.BaseAutoDecoderMaintainer]
    ]:
        def register_decoder(
                _decoder: _typing.Type[base_decoder.BaseAutoDecoderMaintainer]
        ) -> _typing.Type[base_decoder.BaseAutoDecoderMaintainer]:
            if not issubclass(_decoder, base_decoder.BaseAutoDecoderMaintainer):
                raise TypeError
            elif name in cls:
                raise ValueError
            else:
                cls[name] = _decoder
                return _decoder

        return register_decoder

    @classmethod
    def get_decoder(cls, name: str) -> _typing.Type[base_decoder.BaseAutoDecoderMaintainer]:
        if name not in cls:
            raise ValueError(f"Decoder with name \"{name}\" not exist")
        else:
            return cls[name]
