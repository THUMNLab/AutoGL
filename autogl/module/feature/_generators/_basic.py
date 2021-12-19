import torch
import typing as _typing
import autogl
from autogl.data.graph import GeneralStaticGraph
from .._base_feature_engineer import BaseFeatureEngineer
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


class BaseFeatureGenerator(BaseFeatureEngineer):
    def __init__(self, override_features: bool = False):
        super(BaseFeatureGenerator, self).__init__()
        if not isinstance(override_features, bool):
            raise TypeError
        else:
            self._override_features: bool = override_features

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        raise NotImplementedError

    def __transform_homogeneous_static_graph(
            self, homogeneous_static_graph: GeneralStaticGraph
    ) -> GeneralStaticGraph:
        if not (
                homogeneous_static_graph.nodes.is_homogeneous and
                homogeneous_static_graph.edges.is_homogeneous
        ):
            raise ValueError("Provided static graph must be homogeneous")
        if 'x' in homogeneous_static_graph.nodes.data:
            feature_key: _typing.Optional[str] = 'x'
            features: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['x']
            )
        elif 'feat' in homogeneous_static_graph.nodes.data:
            feature_key: _typing.Optional[str] = 'feat'
            features: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['feat']
            )
        else:
            feature_key: _typing.Optional[str] = None
            features: _typing.Optional[torch.Tensor] = None
        if 'y' in homogeneous_static_graph.nodes.data:
            label: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['y']
            )
        elif 'label' in homogeneous_static_graph.nodes.data:
            label: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['label']
            )
        else:
            label: _typing.Optional[torch.Tensor] = None
        if (
                'edge_weight' in homogeneous_static_graph.edges.data and
                homogeneous_static_graph.edges.data['edge_weight'].dim() == 1
        ):
            edge_weight: torch.Tensor = (
                homogeneous_static_graph.edges.data['edge_weight']
            )
        else:
            edge_weight: torch.Tensor = torch.ones(
                homogeneous_static_graph.edges.connections.size(1)
            )
        data = autogl.data.Data(
            edge_index=homogeneous_static_graph.edges.connections,
            x=features, y=label
        )
        setattr(data, "edge_weight", edge_weight)
        extracted_features: torch.Tensor = self._extract_nodes_feature(data)
        if isinstance(feature_key, str):
            nodes_features: torch.Tensor = (
                homogeneous_static_graph.nodes.data[feature_key].view(-1, 1)
                if homogeneous_static_graph.nodes.data[feature_key].dim() == 1
                else homogeneous_static_graph.nodes.data[feature_key]
            )
            assert extracted_features.size(0) == nodes_features.size(0)
            assert extracted_features.dim() == nodes_features.dim() == 2
            homogeneous_static_graph.nodes.data[feature_key] = (
                extracted_features.to(nodes_features.device)
                if self._override_features
                else torch.cat(
                    [nodes_features, extracted_features.to(nodes_features.device)], dim=-1
                )
            )
        else:
            if autogl.backend.DependentBackend.is_pyg():
                homogeneous_static_graph.nodes.data['x'] = extracted_features
            elif autogl.backend.DependentBackend.is_dgl():
                homogeneous_static_graph.nodes.data['feat'] = extracted_features
        return homogeneous_static_graph

    def _transform(
            self, data: _typing.Union[GeneralStaticGraph, _typing.Any]
    ) -> _typing.Union[GeneralStaticGraph, _typing.Any]:
        if isinstance(data, GeneralStaticGraph):
            return self.__transform_homogeneous_static_graph(data)
        else:
            data.x = self._extract_nodes_feature(data)
            return data


@FeatureEngineerUniversalRegistry.register_feature_engineer("OneHot".lower())
class OneHotFeatureGenerator(BaseFeatureGenerator):
    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        num_nodes: int = (
            data.x.size(0)
            if data.x is not None and isinstance(data.x, torch.Tensor)
            else (data.edge_index.max().item() + 1)
        )
        return torch.eye(num_nodes)
