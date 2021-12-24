# copied from torch_scatter
from typing import Optional, Tuple

import torch

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    r"""
    |
    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px
    |
    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.
    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes
    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j
    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.
    .. note::
        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.
    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
    :rtype: :class:`Tensor`
    .. code-block:: python
        from torch_scatter import scatter
        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")
        print(out.size())
    .. code-block::
        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError



def segment_sum_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


def segment_add_csr(src: torch.Tensor, indptr: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_sum_csr(src, indptr, out)


def segment_mean_csr(src: torch.Tensor, indptr: torch.Tensor,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.segment_mean_csr(src, indptr, out)


def segment_min_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_min_csr(src, indptr, out)


def segment_max_csr(
        src: torch.Tensor, indptr: torch.Tensor,
        out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.segment_max_csr(src, indptr, out)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                reduce: str = "sum") -> torch.Tensor:
    r"""
    Reduces all values from the :attr:`src` tensor into :attr:`out` within the
    ranges specified in the :attr:`indptr` tensor along the last dimension of
    :attr:`indptr`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :obj:`indptr.dim() - 1` and by the
    corresponding range index in :attr:`indptr` for dimension
    :obj:`indptr.dim() - 1`.
    The applied reduction is defined via the :attr:`reduce` argument.
    Formally, if :attr:`src` and :attr:`indptr` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_0, ..., x_{m-1}, x_m, x_{m+1}, ..., x_{n-1})` and
    :math:`(x_0, ..., x_{m-2}, y)`, respectively, then :attr:`out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_0, ..., x_{m-2}, y - 1, x_{m}, ..., x_{n-1})`.
    Moreover, the values of :attr:`indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The :attr:`indptr` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes
    .. math::
        \mathrm{out}_i =
        \sum_{j = \mathrm{indptr}[i]}^{\mathrm{indptr}[i+1]-1}~\mathrm{src}_j.
    Due to the use of index pointers, :meth:`segment_csr` is the fastest
    method to apply for grouped reductions.
    .. note::
        In contrast to :meth:`scatter()` and :meth:`segment_coo`, this
        operation is **fully-deterministic**.
    :param src: The source tensor.
    :param indptr: The index pointers between elements to segment.
        The number of dimensions of :attr:`index` needs to be less than or
        equal to :attr:`src`.
    :param out: The destination tensor.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mean"`,
        :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)
    :rtype: :class:`Tensor`
    .. code-block:: python
        from torch_scatter import segment_csr
        src = torch.randn(10, 6, 64)
        indptr = torch.tensor([0, 2, 5, 6])
        indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.
        out = segment_csr(src, indptr, reduce="sum")
        print(out.size())
    .. code-block::
        torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError


def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_csr(src, indptr, out)