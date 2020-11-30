import re

import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos

    @staticmethod
    def from_dict(dictionary):
        r"""Creates a data object from a python dictionary."""
        data = Data()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key, value):
        r"""Returns the dimension in which the attribute :obj:`key` with
        content :obj:`value` gets concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if bool(re.search("(index|face)", key)) else 0

    # own methods for processing
    def get_label_number(self):
        r"""Get the number of labels in this dataset as dict."""
        label_num = {}
        labels = self.y.unique().cpu().detach().numpy().tolist()
        for label in labels:
            label_num[label] = (self.y == label).sum().item()
        return label_num

    def random_splits_mask(self, train_ratio, val_ratio, seed=None):
        r"""If the data has masks for train/val/test, return the splits with specific ratio.

        Parameters
        ----------
        train_ratio : float
            the portion of data that used for training.

        val_ratio : float
            the portion of data that used for validation.

        seed : int
            random seed for splitting dataset.
        """
        rs = torch.get_rng_state()
        rs_cuda = torch.cuda.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        perm = torch.randperm(self.num_nodes)
        train_index = perm[: int(self.num_nodes * train_ratio)]
        val_index = perm[
            int(self.num_nodes * train_ratio) : int(
                self.num_nodes * (train_ratio + val_ratio)
            )
        ]
        test_index = perm[int(self.num_nodes * (train_ratio + val_ratio)) :]
        self.train_mask = index_to_mask(train_index, size=self.num_nodes)
        self.val_mask = index_to_mask(val_index, size=self.num_nodes)
        self.test_mask = index_to_mask(test_index, size=self.num_nodes)

        torch.set_rng_state(rs)
        torch.cuda.set_rng_state(rs_cuda)
        return self

    def random_splits_nodes(self, train_ratio, val_ratio, seed=None):
        r"""If the data uses id of nodes for train/val/test, return the splits with specific ratio.

        Parameters
        ----------
        train_ratio : float
            the portion of data that used for training.

        val_ratio : float
            the portion of data that used for validation.

        seed : int
            random seed for splitting dataset.
        """
        rs = torch.get_rng_state()
        rs_cuda = torch.cuda.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        perm = torch.randperm(self.num_nodes)
        self.train_node = perm[: int(self.num_nodes * train_ratio)]
        self.val_node = perm[
            int(self.num_nodes * train_ratio) : int(
                self.num_nodes * (train_ratio + val_ratio)
            )
        ]
        self.test_node = perm[int(self.num_nodes * (train_ratio + val_ratio)) :]
        self.train_target = self.y[self.train_node]
        self.valid_target = self.y[self.valid_node]
        self.test_target = self.y[self.test_node]

        torch.set_rng_state(rs)
        torch.cuda.set_rng_state(rs_cuda)
        return self

    def random_splits_mask_class(
        self, num_train_per_class, num_val, num_test, seed=None
    ):
        r"""If the data has masks for train/val/test, return the splits with specific number of samples from every class for training.

        Parameters
        ----------
        num_train_per_class : int
            the number of samples from every class used for training.

        num_val : int
            the total number of nodes that used for validation.

        num_test : int
            the total number of nodes that used for testing.

        seed : int
            random seed for splitting dataset.
        """
        rs = torch.get_rng_state()
        rs_cuda = torch.cuda.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        num_classes = self.y.max().cpu().item() + 1
        self.train_mask.fill_(False)
        for c in range(num_classes):
            idx = (self.y == c).nonzero().view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            self.train_mask[idx] = True

        remaining = (~self.train_mask).nonzero().view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        self.val_mask.fill_(False)
        self.val_mask[remaining[:num_val]] = True

        self.test_mask.fill_(False)
        self.test_mask[remaining[num_val : num_val + num_test]] = True

        torch.set_rng_state(rs)
        torch.cuda.set_rng_state(rs_cuda)
        return self

    def random_splits_nodes_class(
        self, num_train_per_class, num_val, num_test, seed=None
    ):
        r"""If the data uses id of nodes for train/val/test, return the splits with specific number of samples from every class for training.

        Parameters
        ----------
        num_train_per_class : int
            the number of samples from every class used for training.

        num_val : int
            the total number of nodes that used for validation.

        num_test : int
            the total number of nodes that used for testing.

        seed : int
            random seed for splitting dataset.
        """
        rs = torch.get_rng_state()
        rs_cuda = torch.cuda.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        num_classes = self.y.max().cpu().item() + 1
        train_mask = torch.zeros(
            self.num_nodes, dtype=torch.bool, device=self.train_node.device
        )
        sup = []
        for c in range(num_classes):
            idx = (self.y == c).nonzero().view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            sup.append(idx)
            train_mask[idx] = True

        self.train_node = torch.cat(sup)

        remaining = (~train_mask).nonzero().view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        self.val_node = remaining[:num_val]
        self.test_node = remaining[num_val : num_val + num_test]

        self.train_target = self.y[self.train_node]
        self.valid_target = self.y[self.valid_node]
        self.test_target = self.y[self.test_node]

        torch.set_rng_state(rs)
        torch.cuda.set_rng_state(rs_cuda)
        return self

    def __inc__(self, key, value):
        r""" "Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_nodes if bool(re.search("(index|face)", key)) else 0

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self("edge_index", "edge_attr"):
            return item.size(self.cat_dim(key, item))
        return None

    @property
    def num_features(self):
        r"""Returns the number of features per node in the graph."""
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.shape[0]
        return torch.max(self.edge_index) + 1

    def is_coalesced(self):
        r"""Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        row, col = self.edge_index
        index = self.num_nodes * row + col
        return row.size(0) == torch.unique(index).size(0)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, :obj:`func` is applied to all present
        attributes.
        """
        for key, item in self(*keys):
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device), *keys)

    def cuda(self, *keys):
        return self.apply(lambda x: x.cuda(), *keys)

    def clone(self):
        return Data.from_dict({k: v.clone() for k, v in self})

    def __repr__(self):
        info = [
            "{}={}".format(key, list(item.size()))
            for key, item in self
            if type(item) != list and type(item) != dict
        ]
        return "{}({})".format(self.__class__.__name__, ", ".join(info))
