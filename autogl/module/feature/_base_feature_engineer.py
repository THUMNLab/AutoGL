import copy
import torch
import typing as _typing
from autogl.data.graph import GeneralStaticGraph
from autogl.data import InMemoryStaticGraphSet

from ...utils import get_logger

LOGGER = get_logger("FeatureEngineer")


class _BaseFeatureEngineer:
    def __and__(self, other):
        raise NotImplementedError

    def fit(
            self, in_memory_static_graph_set: InMemoryStaticGraphSet,
            inplace: bool = True
    ):
        raise NotImplementedError

    def transform(
            self, in_memory_static_graph_set: InMemoryStaticGraphSet,
            inplace: bool = True
    ) -> InMemoryStaticGraphSet:
        raise NotImplementedError


class _ComposedFeatureEngineer(_BaseFeatureEngineer):
    @property
    def fe_components(self) -> _typing.Iterable[_BaseFeatureEngineer]:
        return self.__fe_components

    def __init__(self, feature_engineers: _typing.Iterable[_BaseFeatureEngineer]):
        self.__fe_components: _typing.List[_BaseFeatureEngineer] = []
        for fe in feature_engineers:
            if isinstance(fe, _ComposedFeatureEngineer):
                self.__fe_components.extend(fe.fe_components)
            else:
                self.__fe_components.append(fe)

    def __and__(self, other: _BaseFeatureEngineer):
        return _ComposedFeatureEngineer((self, other))

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


class BaseFeatureEngineer:
    def __init__(self, multi_graph: bool = True, subgraph=False):
        self._multi_graph: bool = multi_graph

    def __and__(self, other):
        return _ComposedFeatureEngineer((self, other))

    @classmethod
    def __reset_graph_set(
            cls, graphs: _typing.Sequence[GeneralStaticGraph],
            in_memory_static_graph_set: InMemoryStaticGraphSet
    ):
        in_memory_static_graph_set.reset_dataset(graphs)

    def _preprocess(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        return static_graph

    def _fit(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        return static_graph

    def _transform(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        return static_graph

    def _postprocess(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        return static_graph

    def fit(
            self, in_memory_static_graph_set: InMemoryStaticGraphSet,
            inplace: bool = True
    ):
        if not inplace:
            in_memory_static_graph_set = copy.deepcopy(in_memory_static_graph_set)
        with torch.no_grad():
            __graphs: _typing.Sequence[GeneralStaticGraph] = [
                self._postprocess(self._transform(self._fit(self._preprocess(g))))
                for g in in_memory_static_graph_set
            ]
            self.__reset_graph_set(__graphs, in_memory_static_graph_set)

    def transform(
            self, in_memory_static_graph_set: InMemoryStaticGraphSet,
            inplace: bool = True
    ) -> InMemoryStaticGraphSet:
        if not inplace:
            in_memory_static_graph_set = copy.deepcopy(in_memory_static_graph_set)
        with torch.no_grad():
            __graphs: _typing.Sequence[GeneralStaticGraph] = [
                self._postprocess(self._transform(self._preprocess(g)))
                for g in in_memory_static_graph_set
            ]
        return in_memory_static_graph_set


class BaseFeature(BaseFeatureEngineer):
    ...
