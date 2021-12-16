import math
import torch
import typing as _typing


def _maybe_num_nodes(edge_index, num_nodes=None):
    if isinstance(num_nodes, int):
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def __coalesce(
        edge_index: torch.Tensor,
        edge_attr: _typing.Union[
            torch.Tensor, _typing.Iterable[torch.Tensor], None
        ] = None,
        num_nodes: _typing.Optional[int] = ...,
        is_sorted: bool = False,
        sort_by_row: bool = True
) -> _typing.Union[
    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor],
    _typing.Tuple[torch.Tensor, _typing.Iterable[torch.Tensor]]
]:
    """
    Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are directly removed, instead of merged.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`Iterable[Tensor]]`)
    """

    if edge_attr is None:
        pass
    elif isinstance(edge_attr, torch.Tensor) and torch.is_tensor(edge_attr):
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError
    elif isinstance(edge_attr, _typing.Iterable):
        if not all([
            (
                    isinstance(attr, torch.Tensor) and
                    attr.size(0) == edge_index.size(1)
            ) for attr in edge_attr
        ]):
            raise ValueError("Invalid edge_attr argument")
    else:
        raise TypeError("Unsupported type of edge_attr argument")

    nnz = edge_index.size(1)
    num_nodes = _maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None and isinstance(edge_attr, torch.Tensor):
            edge_attr = edge_attr[perm]
        elif edge_attr is not None:
            edge_attr = [e[perm] for e in edge_attr]

    mask: _typing.Any = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index if edge_attr is None else (edge_index, edge_attr)

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index
    elif isinstance(edge_attr, torch.Tensor):
        return edge_index, edge_attr[mask]
    elif isinstance(edge_attr, _typing.Iterable):
        return edge_index, [attr[mask] for attr in edge_attr]


def coalesce(
        edge_index: torch.Tensor,
        edge_attr: _typing.Union[
            torch.Tensor, _typing.Iterable[torch.Tensor], None
        ] = None,
        num_nodes: _typing.Optional[int] = ...,
        is_sorted: bool = False,
        sort_by_row: bool = True
) -> _typing.Union[
    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor],
    _typing.Tuple[torch.Tensor, _typing.Iterable[torch.Tensor]]
]:
    """
    Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are directly removed, instead of merged.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`Iterable[Tensor]]`)
    """
    if not isinstance(num_nodes, int):
        num_nodes = None
    try:
        import torch_geometric
        return torch_geometric.utils.coalesce(
            edge_index, edge_attr, num_nodes,
            is_sorted=is_sorted,
            sort_by_row=sort_by_row
        )
    except ModuleNotFoundError:
        return __coalesce(
            edge_index, edge_attr, num_nodes,
            is_sorted=is_sorted,
            sort_by_row=sort_by_row
        )


def to_undirected(
        edge_index: torch.Tensor,
        edge_attr: _typing.Optional[_typing.Union[torch.Tensor, _typing.List[torch.Tensor]]] = None,
        num_nodes: _typing.Optional[int] = ...,
        __reduce: str = "add",
) -> _typing.Union[
    torch.Tensor, _typing.Tuple[torch.Tensor, torch.Tensor],
    _typing.Tuple[torch.Tensor, _typing.List[torch.Tensor]]
]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        __reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if edge_attr is not None and isinstance(edge_attr, torch.Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif edge_attr is not None:
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes)


class _SplitResult:
    def __init__(
            self,
            train_pos_edge_index: torch.Tensor,
            train_pos_edge_attr: _typing.Optional[torch.Tensor],
            train_neg_adj_mask: torch.Tensor,
            val_pos_edge_index: torch.Tensor,
            val_pos_edge_attr: _typing.Optional[torch.Tensor],
            val_neg_edge_index: torch.Tensor,
            test_pos_edge_index: torch.Tensor,
            test_pos_edge_attr: _typing.Optional[torch.Tensor],
            test_neg_edge_index: torch.Tensor
    ):
        self.train_pos_edge_index: torch.Tensor = train_pos_edge_index
        self.train_pos_edge_attr: _typing.Optional[torch.Tensor] = train_pos_edge_attr
        self.train_neg_adj_mask: torch.Tensor = train_neg_adj_mask
        self.val_pos_edge_index: torch.Tensor = val_pos_edge_index
        self.val_pos_edge_attr: _typing.Optional[torch.Tensor] = val_pos_edge_attr
        self.val_neg_edge_index: torch.Tensor = val_neg_edge_index
        self.test_pos_edge_index: torch.Tensor = test_pos_edge_index
        self.test_pos_edge_attr: _typing.Optional[torch.Tensor] = test_pos_edge_attr
        self.test_neg_edge_index: torch.Tensor = test_neg_edge_index


def train_test_split_edges(
        edge_index: torch.Tensor,
        edge_attr: _typing.Optional[_typing.Union[torch.Tensor, _typing.List[torch.Tensor]]] = None,
        num_nodes: _typing.Optional[int] = ...,
        val_ratio: float = 0.05,
        test_ratio: float = 0.1
):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    .. warning::

        :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
        will be removed in a future release.
        Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)
    """
    row, col = edge_index
    num_nodes = _maybe_num_nodes(edge_index, num_nodes)

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        val_pos_edge_attr = edge_attr[:n_v]
    else:
        val_pos_edge_attr = None

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        test_pos_edge_attr = edge_attr[n_v:n_v + n_t]
    else:
        test_pos_edge_attr = None

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        train_pos_edge_index, train_pos_edge_attr = to_undirected(
            train_pos_edge_index, edge_attr[n_v + n_t:]
        )
    else:
        train_pos_edge_index = to_undirected(train_pos_edge_index)
        train_pos_edge_attr = None

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    test_neg_edge_index = torch.stack([row, col], dim=0)

    return _SplitResult(
        train_pos_edge_index, train_pos_edge_attr, train_neg_adj_mask,
        val_pos_edge_index, val_pos_edge_attr, val_neg_edge_index,
        test_pos_edge_index, test_pos_edge_attr, test_neg_edge_index
    )
