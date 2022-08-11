import typing as _typing
import autogl.data.graph


class _AbstractBaseFeatureEngineer:
    def fit(self, dataset):
        raise NotImplementedError

    def transform(self, dataset, inplace: bool = True):
        raise NotImplementedError

    def fit_transform(self, dataset, inplace: bool = True):
        raise NotImplementedError


class _ComposedFeatureEngineer(_AbstractBaseFeatureEngineer):
    @property
    def fe_components(self) -> _typing.Iterable[_AbstractBaseFeatureEngineer]:
        return self.__fe_components

    def fit(self, dataset):
        for fe in self.fe_components:
            dataset = fe.fit(dataset)
        return dataset

    def transform(self, dataset, inplace: bool = True):
        for fe in self.fe_components:
            dataset = fe.transform(dataset, inplace)
        return dataset

    def fit_transform(self, dataset, inplace: bool = True):
        for fe in self.fe_components:
            dataset = fe.fit(dataset)
        for fe in self.fe_components:
            dataset = fe.transform(dataset)
        return dataset

    def __init__(self, feature_engineers: _typing.Iterable[_AbstractBaseFeatureEngineer]):
        self.__fe_components: _typing.List[_AbstractBaseFeatureEngineer] = []
        for fe in feature_engineers:
            if isinstance(fe, _ComposedFeatureEngineer):
                self.__fe_components.extend(fe.fe_components)
            else:
                self.__fe_components.append(fe)

    def __and__(self, other: _AbstractBaseFeatureEngineer):
        return _ComposedFeatureEngineer((self, other))


class _BaseFeatureEngineer(_AbstractBaseFeatureEngineer):
    def __and__(self, other):
        return _ComposedFeatureEngineer((self, other))

    def fit(self, dataset):
        raise NotImplementedError

    def transform(self, dataset, inplace: bool = True):
        raise NotImplementedError

    def fit_transform(self, dataset, inplace: bool = True):
        return self.transform(self.fit(dataset), inplace)

    def _preprocess(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        raise NotImplementedError

    def _postprocess(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        raise NotImplementedError

    def _fit(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        raise NotImplementedError

    def _transform(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        raise NotImplementedError


class BaseFeatureEngineer(_BaseFeatureEngineer):
    def fit(self, dataset):
        raise NotImplementedError

    def transform(self, dataset, inplace: bool = True):
        raise NotImplementedError

    def _preprocess(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        return data

    def _postprocess(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        return data

    def _fit(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        return data

    def _transform(
            self, data: _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[autogl.data.graph.GeneralStaticGraph, _typing.Any]:
        return data
