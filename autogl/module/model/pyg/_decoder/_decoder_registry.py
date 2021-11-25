import typing as _typing
from . import _decoder


class _RepresentationDecoderUniversalRegistryMetaclass(type):
    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        return super(_RepresentationDecoderUniversalRegistryMetaclass, mcs).__new__(
            mcs, name, bases, namespace
        )

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_RepresentationDecoderUniversalRegistryMetaclass, cls).__init__(
            name, bases, namespace
        )
        cls._representation_decoder_registry: _typing.MutableMapping[
            str, _typing.Type[_decoder.RepresentationDecoder]
        ] = {}


class RepresentationDecoderUniversalRegistry(
    metaclass=_RepresentationDecoderUniversalRegistryMetaclass
):
    @classmethod
    def register_representation_decoder(cls, name: str) -> _typing.Callable[
        [_typing.Type[_decoder.RepresentationDecoder]],
        _typing.Type[_decoder.RepresentationDecoder]
    ]:
        def register_decoder(
                decoder: _typing.Type[_decoder.RepresentationDecoder]
        ) -> _typing.Type[_decoder.RepresentationDecoder]:
            if name in cls._representation_decoder_registry:
                raise ValueError(
                    f"Representation Decoder with name \"{name}\" already exists!"
                )
            elif not issubclass(decoder, _decoder.RepresentationDecoder):
                raise TypeError
            else:
                cls._representation_decoder_registry[name] = decoder
                return decoder

        return register_decoder

    @classmethod
    def get_representation_decoder(cls, name: str) -> (
            _typing.Type[_decoder.RepresentationDecoder]
    ):
        decoder: _typing.Optional[
            _typing.Type[_decoder.RepresentationDecoder]
        ] = cls._representation_decoder_registry.get(name)
        if decoder is not None and issubclass(decoder, _decoder.RepresentationDecoder):
            return decoder
        else:
            raise ValueError(f"representation decoder with name \"{name}\" not found")
