import copy
import torch
import typing
import dgl
import autogl.data.graph
import autogl.data.graph.utils.conversion
from . import _data_preprocessor


class DataPreprocessor(_data_preprocessor.DataPreprocessor):
    @classmethod
    def __preprocess(
            cls, data: typing.Union[dgl.DGLGraph, autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        if isinstance(data, dgl.DGLGraph):
            graph = autogl.data.graph.utils.conversion.dgl_graph_to_general_static_graph(data)
            setattr(graph, '__ORIGINAL_TYPE_BEFORE_PREPROCESSING', 'DGLGraph')
            return graph
        else:
            return data

    @classmethod
    def __postprocess(
            cls, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[dgl.DGLGraph, autogl.data.graph.GeneralStaticGraph, typing.Any]:
        if (
                isinstance(data, autogl.data.graph.GeneralStaticGraph) and
                hasattr(data, '__ORIGINAL_TYPE_BEFORE_PREPROCESSING') and
                isinstance(getattr(data, '__ORIGINAL_TYPE_BEFORE_PREPROCESSING', ...), str) and
                getattr(data, '__ORIGINAL_TYPE_BEFORE_PREPROCESSING', ...) == 'DGLGraph'
        ):
            return autogl.data.graph.utils.conversion.general_static_graph_to_dgl_graph(data)
        else:
            return data

    def fit(self, dataset):
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self.__postprocess(self._postprocess(self._fit(self._preprocess(self.__preprocess(data)))))
            return dataset

    def transform(self, dataset, inplace: bool = True):
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self.__postprocess(self._postprocess(self._transform(self._preprocess(self.__preprocess(data)))))
        return dataset

    def fit_transform(self, dataset, inplace: bool = True):
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self.__postprocess(self._postprocess(self._transform(self._fit(self._preprocess(self.__preprocess(data))))))
        return dataset
