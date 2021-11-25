import dgl
import torch
import typing as _typing


class _DecoderDefaultCompatibleTasks(_typing.Container[str], _typing.Iterable[str]):
    """ Default compatible tasks for specific representation decoder """
    def __init__(
            self,
            node_classification_compatible: bool,
            graph_classification_compatible: bool,
            link_prediction_compatible: bool
    ):
        self.__node_classification_compatible: bool = node_classification_compatible
        self.__graph_classification_compatible: bool = graph_classification_compatible
        self.__link_prediction_compatible: bool = link_prediction_compatible

    @property
    def compatible_tasks(self) -> _typing.Sequence[str]:
        compatible_tasks: _typing.MutableSequence[str] = []
        if self.node_classification_compatible:
            compatible_tasks.append("node_classification")
        if self.graph_classification_compatible:
            compatible_tasks.append("graph_classification")
        if self.link_prediction_compatible:
            compatible_tasks.append("link_prediction")
        return tuple(compatible_tasks)

    def __contains__(self, task_identifier: object):
        return (
            task_identifier.lower() in self.compatible_tasks
            if isinstance(task_identifier, str) else False
        )

    def __iter__(self) -> _typing.Iterator[str]:
        return iter(self.compatible_tasks)

    @property
    def node_classification_compatible(self) -> bool:
        return self.__node_classification_compatible

    @property
    def graph_classification_compatible(self) -> bool:
        return self.__graph_classification_compatible

    @property
    def link_prediction_compatible(self) -> bool:
        return self.__link_prediction_compatible


class RepresentationDecoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(RepresentationDecoder, self).__init__()
        self._args: _typing.Sequence[_typing.Any] = args
        self._kwargs: _typing.Mapping[str, _typing.Any] = kwargs
        self.__default_compatible_tasks: _DecoderDefaultCompatibleTasks = (
            _DecoderDefaultCompatibleTasks(
                bool(kwargs.get("node_classification_compatible", False)),
                bool(kwargs.get("graph_classification_compatible", False)),
                bool(kwargs.get("link_classification_compatible", False))
            )
        )

    @property
    def default_compatible_tasks(self) -> _DecoderDefaultCompatibleTasks:
        return self.__default_compatible_tasks

    def __call__(
            self, graph: dgl.DGLGraph,
            features: _typing.Union[torch.Tensor, _typing.Sequence[torch.Tensor]],
            *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
