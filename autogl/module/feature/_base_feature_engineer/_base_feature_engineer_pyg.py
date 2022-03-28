import copy
import torch
import typing as _typing
from autogl.data.graph import GeneralStaticGraph
from . import _base_feature_engineer


class BaseFeatureEngineer(
    _base_feature_engineer.BaseFeatureEngineer
):
    @classmethod
    def __preprocess(
            cls, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        return data  # todo: Support torch_geometric.HeteroData in future

    @classmethod
    def __postprocess(
            cls, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        return data  # todo: Support torch_geometric.HeteroData in future

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
