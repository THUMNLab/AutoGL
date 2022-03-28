import copy
import torch
import typing as _typing
import dgl
from autogl.data.graph import GeneralStaticGraph
import autogl.data.graph.utils.conversion
from . import _base_feature_engineer


class BaseFeatureEngineer(
    _base_feature_engineer.BaseFeatureEngineer
):
    @classmethod
    def __preprocess(
            cls, data: _typing.Union[dgl.DGLGraph, GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        if isinstance(data, dgl.DGLGraph):
            graph = autogl.data.graph.utils.conversion.dgl_graph_to_general_static_graph(data)
            setattr(graph, '_ORIGINAL_TYPE_BEFORE_FE', 'DGLGraph')
            return graph
        else:
            return data

    @classmethod
    def __postprocess(
            cls, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[dgl.DGLGraph, GeneralStaticGraph, _typing.Any]:
        if (
                isinstance(data, GeneralStaticGraph) and
                hasattr(data, '_ORIGINAL_TYPE_BEFORE_FE') and
                isinstance(getattr(data, '_ORIGINAL_TYPE_BEFORE_FE'), str) and
                getattr(data, '_ORIGINAL_TYPE_BEFORE_FE') == 'DGLGraph'
        ):
            return autogl.data.graph.utils.conversion.general_static_graph_to_dgl_graph(data)
        else:
            return data

    def fit(self, dataset):
        dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self.__postprocess(
                    self._postprocess(self._fit(self._preprocess(self.__preprocess(data))))
                )
        return dataset

    def transform(self, dataset, inplace: bool = True):
        if not inplace:
            dataset = copy.deepcopy(dataset)
        with torch.no_grad():
            for i, data in enumerate(dataset):
                dataset[i] = self.__postprocess(
                    self._postprocess(self._transform(self._preprocess(self.__preprocess(data))))
                )
        return dataset
