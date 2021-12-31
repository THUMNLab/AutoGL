import typing as _typing
from autogl.utils import universal_registry
from ._base_feature_engineer import BaseFeatureEngineer


class _FeatureEngineerUniversalRegistryMetaclass(type):
    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        return super(_FeatureEngineerUniversalRegistryMetaclass, mcs).__new__(
            mcs, name, bases, namespace
        )

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_FeatureEngineerUniversalRegistryMetaclass, cls).__init__(
            name, bases, namespace
        )
        cls._feature_engineer_universal_registry: _typing.MutableMapping[
            str, _typing.Type[BaseFeatureEngineer]
        ] = {}


class FeatureEngineerUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_feature_engineer(cls, name: str) -> _typing.Callable[
        [_typing.Type[BaseFeatureEngineer]], _typing.Type[BaseFeatureEngineer]
    ]:
        def register_fe(fe: _typing.Type[BaseFeatureEngineer]) -> _typing.Type[BaseFeatureEngineer]:
            if not issubclass(fe, BaseFeatureEngineer):
                raise TypeError
            else:
                cls[name] = fe
                return fe

        return register_fe

    @classmethod
    def get_feature_engineer(cls, name: str) -> _typing.Type[BaseFeatureEngineer]:
        if name not in cls:
            raise ValueError(f"cannot find feature engineer {name}")
        else:
            return cls[name]


FEATURE_DICT = FeatureEngineerUniversalRegistry
