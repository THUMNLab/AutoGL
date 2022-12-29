import copy
import torch
import typing
import autogl.data.graph


class _AbstractDataPreprocessor:
    def fit(self, dataset):
        raise NotImplementedError

    def transform(self, dataset, inplace: bool = True):
        raise NotImplementedError

    def fit_transform(self, dataset, inplace: bool = True):
        raise NotImplementedError


class _ComposedDataPreprocessor(_AbstractDataPreprocessor):
    def __init__(self, data_preprocessors: typing.Iterable[_AbstractDataPreprocessor]):
        self.__data_preprocessors: typing.MutableSequence[_AbstractDataPreprocessor] = []
        for preprocessor in data_preprocessors:
            if isinstance(preprocessor, _ComposedDataPreprocessor):
                self.__data_preprocessors.extend(preprocessor.__data_preprocessors)
            else:
                self.__data_preprocessors.append(preprocessor)

    def fit(self, dataset):
        for preprocessor in self.__data_preprocessors:
            dataset = preprocessor.fit(dataset)
        return dataset

    def transform(self, dataset, inplace: bool = True):
        for preprocessor in self.__data_preprocessors:
            dataset = preprocessor.transform(dataset, inplace)
        return dataset

    def fit_transform(self, dataset, inplace: bool = True):
        for preprocessor in self.__data_preprocessors:
            dataset = preprocessor.fit_transform(dataset, inplace)
        return dataset

    def __and__(self, other: _AbstractDataPreprocessor):
        return _ComposedDataPreprocessor((self, other))


class _DataPreprocessor(_AbstractDataPreprocessor):
    def __and__(self, other):
        return _ComposedDataPreprocessor((self, other))

    def _preprocess(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        raise NotImplementedError

    def _postprocess(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        raise NotImplementedError

    def _fit(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        raise NotImplementedError

    def _transform(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        raise NotImplementedError

    def fit(self, dataset):
        raise NotImplementedError

    def transform(self, dataset, inplace: bool = True):
        raise NotImplementedError

    def fit_transform(self, dataset, inplace: bool = True):
        raise NotImplementedError


class DataPreprocessor(_DataPreprocessor):
    def fit(self, dataset):
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self._postprocess(self._fit(self._preprocess(data)))
            return dataset

    def transform(self, dataset, inplace: bool = True):
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self._postprocess(self._transform(self._preprocess(data)))
        return dataset

    def fit_transform(self, dataset, inplace: bool = True):
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self._postprocess(self._transform(self._fit(self._preprocess(data))))
        return dataset

    def _preprocess(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        return data

    def _postprocess(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        return data

    def _fit(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        return data

    def _transform(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        return data
