import torch
import typing as _typing
import autogl
from autogl.data.graph import GeneralStaticGraph
from .._base_feature_engineer import BaseFeatureEngineer
from .._feature_engineer_registry import FeatureEngineerUniversalRegistry


class BaseFeatureGenerator(BaseFeatureEngineer):
    def _preprocess(self, static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        if not (
                static_graph.nodes.is_homogeneous and
                static_graph.edges.is_homogeneous
        ):
            raise ValueError("Provided static graph must be homogeneous")
        else:
            return static_graph

    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def __to_data(cls, homogeneous_static_graph: GeneralStaticGraph) -> autogl.data.Data:
        if 'x' in homogeneous_static_graph.nodes.data:
            features: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['x']
            )
        elif 'feat' in homogeneous_static_graph.nodes.data:
            features: _typing.Optional[torch.Tensor] = (
                homogeneous_static_graph.nodes.data['feat']
            )
        else:
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
        return data

    def _transform(self, homogeneous_static_graph: GeneralStaticGraph) -> GeneralStaticGraph:
        nodes_features: torch.Tensor = self._extract_nodes_feature(
            self.__to_data(homogeneous_static_graph)
        )
        if not isinstance(nodes_features, torch.Tensor):
            raise TypeError
        elif nodes_features.dim() == 0:
            raise ValueError
        elif nodes_features.dim() == 1:
            nodes_features = nodes_features.view(-1, 1)
        if 'x' in homogeneous_static_graph.nodes.data:
            x: torch.Tensor = (
                homogeneous_static_graph.nodes.data['x'].view(-1, 1)
                if homogeneous_static_graph.nodes.data['x'].dim() == 1
                else homogeneous_static_graph.nodes.data['x']
            )
            assert nodes_features.size(0) == x.size(0)
            assert nodes_features.dim() == x.dim() == 2
            homogeneous_static_graph.nodes.data['x'] = torch.cat(
                [x, nodes_features.to(x.dtype)], dim=-1
            )
        elif 'feat' in homogeneous_static_graph.nodes.data:
            x: torch.Tensor = (
                homogeneous_static_graph.nodes.data['feat'].view(-1, 1)
                if homogeneous_static_graph.nodes.data['feat'].dim() == 1
                else homogeneous_static_graph.nodes.data['feat']
            )
            assert nodes_features.size(0) == x.size(0)
            assert nodes_features.dim() == x.dim() == 2
            homogeneous_static_graph.nodes.data['feat'] = torch.cat(
                [x, nodes_features.to(x.dtype)], dim=-1
            )
        else:
            if autogl.backend.DependentBackend.is_pyg():
                homogeneous_static_graph.nodes.data['x'] = nodes_features
            elif autogl.backend.DependentBackend.is_dgl():
                homogeneous_static_graph.nodes.data['feat'] = nodes_features
        return homogeneous_static_graph


@FeatureEngineerUniversalRegistry.register_feature_engineer("OneHot".lower())
class OneHotFeatureGenerator(BaseFeatureGenerator):
    def _extract_nodes_feature(self, data: autogl.data.Data) -> torch.Tensor:
        num_nodes: int = (
            data.x.size(0)
            if data.x is not None and isinstance(data.x, torch.Tensor)
            else (data.edge_index.max().item() + 1)
        )
        return torch.eye(num_nodes)
