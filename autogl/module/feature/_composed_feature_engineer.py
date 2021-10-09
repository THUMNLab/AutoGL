import typing as _typing
from . import _base_feature_engineer


class _ComposedFeatureEngineer(_base_feature_engineer.BaseFeatureEngineer):
    ...


class ComposedFeatureEngineer(_ComposedFeatureEngineer):
    @property
    def fe_components(self) -> _typing.Iterable[_base_feature_engineer.BaseFeatureEngineer]:
        raise NotImplementedError  # todo

    def __init__(self, feature_engineers: _typing.Iterable[_base_feature_engineer.BaseFeatureEngineer]):
        super(ComposedFeatureEngineer, self).__init__()
        self.__fe_components: _typing.List[_base_feature_engineer.BaseFeatureEngineer] = []
        for fe in feature_engineers:
            if isinstance(fe, ComposedFeatureEngineer):
                self.__fe_components.extend(fe.fe_components)
            elif isinstance(fe, _base_feature_engineer.BaseFeatureEngineer):
                self.__fe_components.append(fe)
            else:
                raise TypeError

    def fit(self, in_memory_static_graph_set, inplace: bool = True):
        for fe in self.fe_components:
            fe.fit(in_memory_static_graph_set, inplace)

    def transform(
            self, in_memory_static_graph_set,
            inplace: bool = True
    ):
        for fe in self.fe_components:
            in_memory_static_graph_set = fe.transform(
                in_memory_static_graph_set, inplace
            )
        return in_memory_static_graph_set
