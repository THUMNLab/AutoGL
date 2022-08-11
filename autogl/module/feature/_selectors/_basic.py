import numpy as np
import torch
import typing as _typing
from autogl.data.graph import GeneralStaticGraph
from .._base_feature_engineer import BaseFeatureEngineer
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


class BaseFeatureSelector(BaseFeatureEngineer):
    def __init__(self):
        self._selection : _typing.Optional[torch.Tensor] = None
        super(BaseFeatureSelector, self).__init__()

    def __transform_homogeneous_static_graph(
            self, static_graph: GeneralStaticGraph
    ) -> GeneralStaticGraph:
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
            self, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        if isinstance(data, GeneralStaticGraph):
            return self.__transform_homogeneous_static_graph(data)
        elif (
                hasattr(data, 'x') and isinstance(data.x, torch.Tensor) and
                torch.is_tensor(data.x) and data.x.dim() == 2
        ):
            data.x = data.x[:, self._selection]
            return data
        else:
            return data


@FeatureEngineerUniversalRegistry.register_feature_engineer("FilterConstant")
class FilterConstant(BaseFeatureSelector):
    r"""drop constant features"""

    def _fit(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        if (
                'x' in static_graph.nodes.data and
                self._selection not in (Ellipsis, None) and
                isinstance(self._selection, torch.Tensor) and
                torch.is_tensor(self._selection) and self._selection.dim() == 1
        ):
            feature: _typing.Optional[np.ndarray] = static_graph.nodes.data['x'].numpy()
        elif (
                'feat' in static_graph.nodes.data and
                self._selection not in (Ellipsis, None) and
                isinstance(self._selection, torch.Tensor) and
                torch.is_tensor(self._selection) and self._selection.dim() == 1
        ):
            feature: _typing.Optional[np.ndarray] = static_graph.nodes.data['feat'].numpy()
        else:
            feature: _typing.Optional[np.ndarray] = None
        self._selection: _typing.Optional[torch.Tensor] = torch.from_numpy(
            np.where(np.all(feature == feature[0, :], axis=0) == np.array(False))[0]
            if feature is not None and isinstance(feature, np.ndarray) and feature.ndim == 2
            else None
        )
        return static_graph
