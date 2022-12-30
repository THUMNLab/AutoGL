import numpy as np
import torch
import typing
import autogl.data.graph
from .._feature_engineer import FeatureEngineer
from ..._data_preprocessor_registry import DataPreprocessorUniversalRegistry


class BaseFeatureSelector(FeatureEngineer):
    def __init__(self):
        super(BaseFeatureSelector, self).__init__()
        self._selection: typing.Optional[torch.Tensor] = None

    def __transform_homogeneous_static_graph(
            self, static_graph: autogl.data.graph.GeneralStaticGraph
    ) -> autogl.data.graph.GeneralStaticGraph:
        if (
                'x' in static_graph.nodes.data and
                isinstance(self._selection, (torch.Tensor, np.ndarray))
        ):
            static_graph.nodes.data['x'] = static_graph.nodes.data['x'][:, self._selection]
        if (
                'feat' in static_graph.nodes.data and
                isinstance(self._selection, (torch.Tensor, np.ndarray))
        ):
            static_graph.nodes.data['feat'] = static_graph.nodes.data['feat'][:, self._selection]
        return static_graph

    def _transform(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        if isinstance(data, autogl.data.graph.GeneralStaticGraph):
            return self.__transform_homogeneous_static_graph(data)
        elif (
                hasattr(data, 'x') and isinstance(data.x, torch.Tensor) and
                torch.is_tensor(data.x) and data.x.dim() == 2
        ):
            data.x = data.x[:, self._selection]
            return data
        else:
            return data


@DataPreprocessorUniversalRegistry.register_data_preprocessor("FilterConstant")
class FilterConstant(BaseFeatureSelector):
    r"""drop constant features"""

    def _fit(
            self, data: typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]
    ) -> typing.Union[autogl.data.graph.GeneralStaticGraph, typing.Any]:
        if isinstance(data, autogl.data.graph.GeneralStaticGraph):
            if 'x' in data.nodes.data:
                feature: typing.Optional[np.ndarray] = data.nodes.data['x'].numpy()
            elif 'feat' in data.nodes.data:
                feature: typing.Optional[np.ndarray] = data.nodes.data['feat'].numpy()
            else:
                feature: typing.Optional[np.ndarray] = None
        elif (
                hasattr(data, 'x') and isinstance(data.x, torch.Tensor) and
                torch.is_tensor(data.x) and data.x.dim() == 2
        ):
            feature: typing.Optional[np.ndarray] = data.x.numpy()
        else:
            feature: typing.Optional[np.ndarray] = None

        if feature is not None and isinstance(feature, np.ndarray) and feature.ndim == 2:
            self._selection: typing.Optional[torch.Tensor] = torch.from_numpy(
                np.where(np.all(feature == feature[0, :], axis=0) == np.array(False))[0]
            )
        else:
            self._selection: typing.Optional[torch.Tensor] = None
        return data
