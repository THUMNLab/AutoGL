from . import _abstract_views


class GeneralStaticGraph:
    @property
    def nodes(self) -> _abstract_views.HeterogeneousNodeView:
        raise NotImplementedError

    @property
    def edges(self) -> _abstract_views.HeterogeneousEdgesView:
        raise NotImplementedError

    @property
    def data(self) -> _abstract_views.GraphDataView:
        raise NotImplementedError
