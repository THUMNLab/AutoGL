import torch
import typing
from autogl.utils import universal_registry


class GraphSampler(torch.nn.Module, typing.Iterable):
    def __iter__(self):
        raise NotImplementedError


class SampledSubgraph:
    ...


class GraphSamplerUniversalRegistry(universal_registry.UniversalRegistryBase):
    @classmethod
    def register_graph_sampler(cls, name: str) -> typing.Callable[
        [typing.Type[GraphSampler]], typing.Type[GraphSampler]
    ]:
        def register_sampler(
                graph_sampler: typing.Type[GraphSampler]
        ) -> typing.Type[GraphSampler]:
            if not issubclass(graph_sampler, GraphSampler):
                raise TypeError
            else:
                cls[name] = graph_sampler
                return graph_sampler

        return register_sampler

    @classmethod
    def get_graph_sampler(cls, name: str) -> typing.Type[GraphSampler]:
        if name not in cls:
            raise ValueError(f"Graph Sampler with name \"{name}\" not exist")
        else:
            return cls[name]


def instantiate_graph_sampler(
        graph_sampler_name: str, data, sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
) -> GraphSampler:
    return GraphSamplerUniversalRegistry[graph_sampler_name](data, sampler_configurations, **kwargs)
