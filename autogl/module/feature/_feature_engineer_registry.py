import typing as _typing

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


class FeatureEngineerUniversalRegistry(metaclass=_FeatureEngineerUniversalRegistryMetaclass):
    @classmethod
    def register_feature_engineer(cls, name: str) -> _typing.Callable[
        [_typing.Type[BaseFeatureEngineer]], _typing.Type[BaseFeatureEngineer]
    ]:
        def register_fe(
                fe: _typing.Type[BaseFeatureEngineer]
        ) -> _typing.Type[BaseFeatureEngineer]:
            if name in cls._feature_engineer_universal_registry:
                raise ValueError(
                    f"Feature Engineer with name \"{name}\" already exists!"
                )
            elif not issubclass(fe, BaseFeatureEngineer):
                raise TypeError
            else:
                cls._feature_engineer_universal_registry[name] = fe
                return fe
        return register_fe

    @classmethod
    def get_feature_engineer(cls, name: str) -> _typing.Type[BaseFeatureEngineer]:
        if name in cls._feature_engineer_universal_registry:
            return cls._feature_engineer_universal_registry[name]
        else:
            raise ValueError(f"cannot find feature engineer {name}")


class _DeprecatedFeatureDict:
    def __contains__(self, name: str) -> bool:
        return name in FeatureEngineerUniversalRegistry._feature_engineer_universal_registry

    def __getitem__(self, name: str) -> _typing.Type[BaseFeatureEngineer]:
        return FeatureEngineerUniversalRegistry.get_feature_engineer(name)


FEATURE_DICT = _DeprecatedFeatureDict()
