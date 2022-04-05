import torch
import typing
import torch_geometric.loader
from .. import _graph_sampler, _sampler_utility


class PyGGraphSampler(_graph_sampler.GraphSampler):
    def __iter__(self):
        raise NotImplementedError


class PyGHomogeneousGraphSampler(PyGGraphSampler):
    def __iter__(self):
        raise NotImplementedError


class PyGSampledSubgraph(_graph_sampler.SampledSubgraph):
    @property
    def data(self) -> torch_geometric.data.Data:
        raise NotImplementedError


class _PyGSampledHomogeneousSubgraph(PyGSampledSubgraph):
    @property
    def data(self) -> torch_geometric.data.Data:
        return self._data

    def __init__(self, data: torch_geometric.data.Data, *_args, **_kwargs):
        if not isinstance(data, torch_geometric.data.Data):
            raise TypeError
        self._data: torch_geometric.data.Data = data


class _PyGHomogeneousGraphSamplerIterator(typing.Iterator):
    def __init__(
            self, iterable: typing.Iterable[torch_geometric.data.Data],
            transform: typing.Optional[typing.Callable[[torch_geometric.data.Data], typing.Any]] = ...
    ):
        self.__iterator: typing.Iterator[torch_geometric.data.Data] = iter(iterable)
        self._transform: typing.Optional[typing.Callable[[torch_geometric.data.Data], typing.Any]] = (
            transform if transform is not None and transform is not Ellipsis and callable(transform) else None
        )

    def __iter__(self) -> '_PyGHomogeneousGraphSamplerIterator':
        return self

    def __next__(self):
        __data: torch_geometric.data.Data = next(self.__iterator)
        return self._transform(__data) if self._transform is not None and callable(self._transform) else __data


@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('neighbor_sampler')
@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('pyg_neighbor_sampler')
class PyGNeighborSampler(PyGHomogeneousGraphSampler):
    def __init__(
            self, data: torch_geometric.data.Data,
            sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
    ):
        super(PyGNeighborSampler, self).__init__()
        __filtered_configurations, remaining_configurations = _sampler_utility.ConfigurationsFilter(
            (
                (
                    ('num_neighbors', 'sizes', 'FanOuts'.lower()),
                    lambda num_neighbors: isinstance(num_neighbors, typing.Iterable) and all(
                        (isinstance(_num_neighbors, int) and (_num_neighbors == -1 or _num_neighbors > 0))
                        for _num_neighbors in num_neighbors
                    ),
                    lambda num_neighbors: list(num_neighbors),
                    None, f"specified num_neighbors/sizes/{'FanOuts'.lower()} argument must be list of integer"
                ),
                (
                    ('input_nodes', 'node_idx', 'target_nodes'),
                    lambda input_nodes: input_nodes is None or isinstance(input_nodes, torch.Tensor), None,
                    ..., "specified input_nodes/node_idx/target_nodes argument must be either None or Tensor"
                ),
                (('replace',), ..., lambda replace: bool(replace), ..., None),
                (('directed',), ..., lambda directed: bool(directed), ..., None),
                (
                    ('batch_size',), lambda batch_size: isinstance(batch_size, int) and batch_size > 0,
                    lambda batch_size: int(batch_size), ..., None
                ),
                (('shuffle',), ..., lambda shuffle: bool(shuffle), ..., None),
                (
                    ('transform',),
                    lambda _transform: _transform is None or _transform is Ellipsis or callable(_transform),
                    lambda _transform: _transform if callable(_transform) else None,
                    ..., 'specified transform argument must be either None or callable transform function'
                )
            )
        ).filter({**sampler_configurations, **kwargs})
        _filtered_configurations: typing.MutableMapping[str, typing.Any] = dict(__filtered_configurations)
        _transform: typing.Optional[typing.Callable[[torch_geometric.data.Data], torch_geometric.data.Data]] = (
            _filtered_configurations.pop('transform', None)
        )

        def transform(__data: torch_geometric.data.Data) -> torch_geometric.data.Data:
            if not hasattr(__data, 'batch_size'):
                raise ValueError
            if not isinstance(__data.batch_size, int) and __data.batch_size > 0:
                raise ValueError
            __data.target_nodes_index = torch.arange(0, __data.batch_size, device=__data.edge_index.device)
            return _transform(__data) if _transform is not None and callable(_transform) else __data

        self._neighbor_loader: torch_geometric.loader.NeighborLoader = torch_geometric.loader.NeighborLoader(
            data, **{**_filtered_configurations, **remaining_configurations}, transform=transform
        )

    def __iter__(self) -> typing.Iterator[_PyGSampledHomogeneousSubgraph]:
        return _PyGHomogeneousGraphSamplerIterator(
            self._neighbor_loader, lambda data: _PyGSampledHomogeneousSubgraph(data)
        )


@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('graph_saint_node_sampler')
@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('pyg_graph_saint_node_sampler')
class PyGGraphSAINTNodeSampler(PyGHomogeneousGraphSampler):
    def __init__(
            self, data: torch_geometric.data.Data,
            sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
    ):
        super(PyGGraphSAINTNodeSampler, self).__init__()
        _filtered_configurations, _remaining_configurations = _sampler_utility.ConfigurationsFilter(
            (
                (
                    ('batch_size',), lambda batch_size: isinstance(batch_size, int) and batch_size > 0, ..., None,
                    "specified batch_size argument MUST be a positive integer "
                    "representing the approximate number of samples per batch"
                ),
                (
                    ('num_steps', 'num_iterations'), lambda num_steps: isinstance(num_steps, int) and num_steps > 0,
                    ..., ...,
                    "specified num_steps/num_iterations argument MUST be a positive integer "
                    "representing the number of iterations per epoch"
                ),
                (
                    ('sample_coverage',), lambda sample_coverage: isinstance(sample_coverage, int) and sample_coverage >= 0,
                    ..., ...,
                    "specified sample_coverage argument MUST be a non-negative argument "
                    "representing the coverage factor should be used to compute normalization statistics"
                ),
                (
                    ('save_dir',), lambda save_dir: save_dir in (Ellipsis, None) or isinstance(save_dir, str),
                    lambda save_dir: save_dir if isinstance(save_dir, str) else None, ...,
                    'specified save_dir argument must be None or str representing the path of directory '
                    'to save the normalization statistics for faster re-use'
                ),
                (
                    ('log',), lambda _log: isinstance(_log, bool), lambda _log: bool(_log), ...,
                    "specified log argument MUST be a bool representing whether logging any pre-processing progress"
                )
            )
        ).filter({**sampler_configurations, **kwargs})
        self._graph_saint_sampler: torch_geometric.loader.GraphSAINTSampler = (
            torch_geometric.loader.GraphSAINTNodeSampler(
                data, **{**_filtered_configurations, **_remaining_configurations}
            )
        )

    def __iter__(self) -> typing.Iterator[_PyGSampledHomogeneousSubgraph]:
        return _PyGHomogeneousGraphSamplerIterator(
            self._graph_saint_sampler, lambda data: _PyGSampledHomogeneousSubgraph(data)
        )


@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('graph_saint_edge_sampler')
@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('pyg_graph_saint_edge_sampler')
class PyGGraphSAINTEdgeSampler(PyGHomogeneousGraphSampler):
    def __init__(
            self, data: torch_geometric.data.Data,
            sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
    ):
        super(PyGGraphSAINTEdgeSampler, self).__init__()
        _filtered_configurations, _remaining_configurations = _sampler_utility.ConfigurationsFilter(
            (
                (
                    ('batch_size',), lambda batch_size: isinstance(batch_size, int) and batch_size > 0, ..., None,
                    "specified batch_size argument MUST be a positive integer "
                    "representing the approximate number of samples per batch"
                ),
                (
                    ('num_steps', 'num_iterations'), lambda num_steps: isinstance(num_steps, int) and num_steps > 0,
                    ..., ...,
                    "specified num_steps/num_iterations argument MUST be a positive integer "
                    "representing the number of iterations per epoch"
                ),
                (
                    ('sample_coverage',), lambda sample_coverage: isinstance(sample_coverage, int) and sample_coverage >= 0,
                    ..., ...,
                    "specified sample_coverage argument MUST be a non-negative argument "
                    "representing the coverage factor should be used to compute normalization statistics"
                ),
                (
                    ('save_dir',), lambda save_dir: save_dir in (Ellipsis, None) or isinstance(save_dir, str),
                    lambda save_dir: save_dir if isinstance(save_dir, str) else None, ...,
                    'specified save_dir argument must be None or str representing the path of directory '
                    'to save the normalization statistics for faster re-use'
                ),
                (
                    ('log',), lambda _log: isinstance(_log, bool), lambda _log: bool(_log), ...,
                    "specified log argument MUST be a bool representing whether logging any pre-processing progress"
                )
            )
        ).filter({**sampler_configurations, **kwargs})
        self._graph_saint_sampler: torch_geometric.loader.GraphSAINTSampler = (
            torch_geometric.loader.GraphSAINTEdgeSampler(
                data, **{**_filtered_configurations, **_remaining_configurations}
            )
        )

    def __iter__(self) -> typing.Iterator[_PyGSampledHomogeneousSubgraph]:
        return _PyGHomogeneousGraphSamplerIterator(
            self._graph_saint_sampler, lambda data: _PyGSampledHomogeneousSubgraph(data)
        )


@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('graph_saint_random_walk_sampler')
@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('pyg_graph_saint_random_walk_sampler')
class PyGGraphSAINTRandomWalkSampler(PyGHomogeneousGraphSampler):
    def __init__(
            self, data: torch_geometric.data.Data,
            sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
    ):
        super(PyGGraphSAINTRandomWalkSampler, self).__init__()
        _filtered_configurations, _remaining_configurations = _sampler_utility.ConfigurationsFilter(
            (
                (
                    ('batch_size',), lambda batch_size: isinstance(batch_size, int) and batch_size > 0, ..., None,
                    "specified batch_size argument MUST be a positive integer "
                    "representing the approximate number of samples per batch"
                ),
                (
                    ('walk_length',), lambda walk_length: isinstance(walk_length, int) and walk_length > 0, ..., None,
                    "specified walk_length argument MUST be a positive integer "
                    "representing the length of each random walk"
                ),
                (
                    ('num_steps', 'num_iterations'), lambda num_steps: isinstance(num_steps, int) and num_steps > 0,
                    ..., ...,
                    "specified num_steps/num_iterations argument MUST be a positive integer "
                    "representing the number of iterations per epoch"
                ),
                (
                    ('sample_coverage',), lambda s_coverage: isinstance(s_coverage, int) and s_coverage >= 0, ..., ...,
                    "specified sample_coverage argument MUST be a non-negative argument "
                    "representing the coverage factor should be used to compute normalization statistics"
                ),
                (
                    ('save_dir',), lambda save_dir: save_dir in (Ellipsis, None) or isinstance(save_dir, str),
                    lambda save_dir: save_dir if isinstance(save_dir, str) else None, ...,
                    'specified save_dir argument must be None or str representing the path of directory '
                    'to save the normalization statistics for faster re-use'
                ),
                (
                    ('log',), lambda _log: isinstance(_log, bool), lambda _log: bool(_log), ...,
                    "specified log argument MUST be a bool representing whether logging any pre-processing progress"
                )
            )
        ).filter({**sampler_configurations, **kwargs})
        self._graph_saint_sampler: torch_geometric.loader.GraphSAINTSampler = (
            torch_geometric.loader.GraphSAINTRandomWalkSampler(
                data, **{**_filtered_configurations, **_remaining_configurations}
            )
        )

    def __iter__(self) -> typing.Iterator[_PyGSampledHomogeneousSubgraph]:
        return _PyGHomogeneousGraphSamplerIterator(
            self._graph_saint_sampler, lambda data: _PyGSampledHomogeneousSubgraph(data)
        )


@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('cluster_sampler')
@_graph_sampler.GraphSamplerUniversalRegistry.register_graph_sampler('pyg_cluster_sampler')
class PyGClusterSampler(PyGHomogeneousGraphSampler):
    def __init__(
            self, data: torch_geometric.data.Data,
            sampler_configurations: typing.Mapping[str, typing.Any], **kwargs
    ):
        super(PyGClusterSampler, self).__init__()
        _filtered_configurations, _remaining_configurations = _sampler_utility.ConfigurationsFilter(
            (
                (
                    ('num_parts',),
                    lambda num_parts: isinstance(num_parts, int) and num_parts > 0, lambda num_parts: int(num_parts),
                    None, 'specified num_parts argument be positive integer representing the number of partitions'
                ),
                (
                    ('recursive',), lambda recursive: isinstance(recursive, bool), lambda recursive: bool(recursive),
                    ...,
                    'specified recursive argument must be bool '
                    'indicating whether to use multilevel recursive bisection instead of multilevel k-way partitioning'
                ),
                (
                    ('save_dir',), lambda save_dir: save_dir in (Ellipsis, None) or isinstance(save_dir, str),
                    lambda save_dir: save_dir if isinstance(save_dir, str) else None, ...,
                    'specified save_dir argument must be None or str representing the path of directory '
                    'to save the partitioned data for faster re-use'
                ),
                (
                    ('log',), lambda _log: isinstance(_log, bool), lambda _log: bool(_log), ...,
                    "specified log argument MUST be a bool representing whether logging any pre-processing progress"
                )
            )
        ).filter({**sampler_configurations, **kwargs})
        self.__cluster_loader: torch_geometric.loader.ClusterLoader = torch_geometric.loader.ClusterLoader(
            torch_geometric.loader.ClusterData(data, **_filtered_configurations), **_remaining_configurations
        )

    def __iter__(self) -> typing.Iterator[_PyGSampledHomogeneousSubgraph]:
        return _PyGHomogeneousGraphSamplerIterator(
            self.__cluster_loader, lambda data: _PyGSampledHomogeneousSubgraph(data)
        )
