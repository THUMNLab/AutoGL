import copy
import logging
import torch
import typing as _typing
from autogl.data import Dataset

LOGGER = logging.getLogger("FeatureEngineer")


class _BaseFeatureEngineer:
    def __and__(self, other):
        raise NotImplementedError

    def fit_transform(self, dataset: Dataset, inplace=True) -> Dataset:
        """
        Fit and transform dataset inplace or not w.r.t bool argument ``inplace``
        """
        dataset = self.fit(dataset)
        return self.transform(dataset, inplace=inplace)

    def fit(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError

    def transform(self, dataset: Dataset, inplace: bool = True) -> Dataset:
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

    def fit(self, dataset) -> Dataset:
        for fe in self.fe_components:
            dataset = fe.fit(dataset)
        return dataset

    def transform(self, dataset: Dataset, inplace: bool = True) -> Dataset:
        for fe in self.fe_components:
            dataset = fe.transform(dataset, inplace)
        return dataset


class BaseFeature(_BaseFeatureEngineer):
    def __init__(self, multi_graph: bool = True, subgraph=False):
        self._multi_graph: bool = multi_graph

    def __and__(self, other):
        return _ComposedFeatureEngineer((self, other))

    def _preprocess(self, data: _typing.Any) -> _typing.Any:
        return data

    def _fit(self, data: _typing.Any) -> _typing.Any:
        return data

    def _transform(self, data: _typing.Any) -> _typing.Any:
        return data

    def _postprocess(self, data: _typing.Any) -> _typing.Any:
        return data

    def fit(self, dataset: Dataset) -> Dataset:
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self._postprocess(self._transform(self._fit(self._preprocess(data))))
            return dataset

    def transform(self, dataset: Dataset, inplace: bool = True) -> Dataset:
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self._postprocess(self._transform(self._preprocess(data)))
            return dataset


class BaseFeatureEngineer(BaseFeature):
    ...
