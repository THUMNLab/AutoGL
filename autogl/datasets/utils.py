from pdb import set_trace
import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import StratifiedKFold, KFold


def split_edges(dataset, train_ratio, val_ratio):
    datas = [data for data in dataset]
    for i in range(len(datas)):
        datas[i] = train_test_split_edges(
            datas[i], val_ratio, 1 - train_ratio - val_ratio
        )
    dataset.data, dataset.slices = dataset.collate(datas)


def get_label_number(dataset):
    r"""Get the number of labels in this dataset as dict."""
    label_num = {}
    labels = dataset.data.y.unique().cpu().detach().numpy().tolist()
    for label in labels:
        label_num[label] = (dataset.data.y == label).sum().item()
    return label_num


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_splits_mask(dataset, train_ratio=0.2, val_ratio=0.4, seed=None):
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

    assert (
        train_ratio + val_ratio <= 1
    ), "the sum of train_ratio and val_ratio is larger than 1"
    _dataset = [d for d in dataset]
    for data in _dataset:
        r_s = torch.get_rng_state()
        if torch.cuda.is_available():
            r_s_cuda = torch.cuda.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        perm = torch.randperm(data.num_nodes)
        train_index = perm[: int(data.num_nodes * train_ratio)]
        val_index = perm[
            int(data.num_nodes * train_ratio) : int(
                data.num_nodes * (train_ratio + val_ratio)
            )
        ]
        test_index = perm[int(data.num_nodes * (train_ratio + val_ratio)) :]
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

        torch.set_rng_state(r_s)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(r_s_cuda)

    dataset.data, dataset.slices = dataset.collate(_dataset)
    if hasattr(dataset, "__data_list__"):
        delattr(dataset, "__data_list__")
    # while type(dataset.data.num_nodes) == list:
    #    dataset.data.num_nodes = dataset.data.num_nodes[0]
    # dataset.data.num_nodes = dataset.data.num_nodes[0]
    return dataset


def random_splits_mask_class(
    dataset,
    num_train_per_class=20,
    num_val_per_class=30,
    num_val=None,
    num_test=None,
    seed=None,
):
    r"""If the data has masks for train/val/test, return the splits with specific number of samples from every class for training as suggested in Pitfalls of graph neural network evaluation [#]_ for semi-supervised learning.

    References
    ----------
    .. [#] Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018).
        Pitfalls of graph neural network evaluation.
        arXiv preprint arXiv:1811.05868.

    Parameters
    ----------
    num_train_per_class : int
        the number of samples from every class used for training.

    num_val_per_class : int
        the number of samples from every class used for validation.

    num_val : int
        the total number of nodes that used for validation as alternative.

    num_test : int
        the total number of nodes that used for testing as alternative. The rest of the data will be seleted as test set if num_test set to None.

    seed : int
        random seed for splitting dataset.
    """
    data = dataset[0]

    r_s = torch.get_rng_state()
    if torch.cuda.is_available():
        r_s_cuda = torch.cuda.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    num_classes = data.y.max().cpu().item() + 1
    try:
        data.train_mask.fill_(False)
        data.val_mask.fill_(False)
        data.test_mask.fill_(False)
    except:
        train_mask = torch.zeros(
            data.num_nodes, dtype=torch.bool, device=data.edge_index.device
        )
        val_mask = torch.zeros(
            data.num_nodes, dtype=torch.bool, device=data.edge_index.device
        )
        test_mask = torch.zeros(
            data.num_nodes, dtype=torch.bool, device=data.edge_index.device
        )
        setattr(data, "train_mask", train_mask)
        setattr(data, "val_mask", val_mask)
        setattr(data, "test_mask", test_mask)
    for c_i in range(num_classes):
        idx = (data.y == c_i).nonzero().view(-1)
        assert num_train_per_class + num_val_per_class < idx.size(0), (
            "the total number of samples from every class used for training and validation is larger than the total samples in class "
            + str(c_i)
        )
        idx_idx_rand = torch.randperm(idx.size(0))
        idx_train = idx[idx_idx_rand[:num_train_per_class]]
        idx_val = idx[
            idx_idx_rand[num_train_per_class : num_train_per_class + num_val_per_class]
        ]
        data.train_mask[idx_train] = True
        data.val_mask[idx_val] = True

    if num_val is not None:
        remaining = (~data.train_mask).nonzero().view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]
        data.val_mask[remaining[:num_val]] = True
        if num_test is not None:
            data.test_mask[remaining[num_val : num_val + num_test]] = True
        else:
            data.test_mask[remaining[num_val:]] = True
    else:
        remaining = (~(data.train_mask + data.val_mask)).nonzero().view(-1)
        data.test_mask[remaining] = True

    torch.set_rng_state(r_s)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(r_s_cuda)

    datalist = []
    for d in dataset:
        setattr(d, "train_mask", data.train_mask)
        setattr(d, "val_mask", data.val_mask)
        setattr(d, "test_mask", data.test_mask)
        datalist.append(d)
    dataset.data, dataset.slices = dataset.collate(datalist)
    if hasattr(dataset, "__data_list__"):
        delattr(dataset, "__data_list__")
    # while type(dataset.data.num_nodes) == list:
    #     dataset.data.num_nodes = dataset.data.num_nodes[0]
    # dataset.data.num_nodes = dataset.data.num_nodes[0]
    return dataset


def graph_cross_validation(
    dataset, n_splits=10, shuffle=True, random_seed=42, stratify=False
):
    r"""Cross validation for graph classification data, returning one fold with specific idx in autogl.datasets or pyg.Dataloader(default)

    Parameters
    ----------
    dataset : str
        dataset with multiple graphs.

    n_splits : int
        the number of how many folds will be splitted.

    shuffle : bool
        shuffle or not for sklearn.model_selection.StratifiedKFold

    random_seed : int
        random_state for sklearn.model_selection.StratifiedKFold
    """
    if stratify:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_seed
        )
    else:
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    idx_list = []

    # BUG: from pytorch_geometric, not sure whether it is a bug. The dataset.data will return
    # the data of original dataset even if the input dataset is subset of original. We hackfix
    # this bug currently by iterating y.

    dataset_y = [data.y[0].tolist() for data in dataset]

    for idx in skf.split(np.zeros(len(dataset_y)), dataset_y):
        idx_list.append(idx)
    dataset.idx_list = idx_list
    dataset.n_splits = n_splits
    # BUG: only saving idx will result in different references when calling multiple times,
    # we need to also save splits in advance.
    dataset.cv_dict = [
        {
            "train": dataset[dataset.idx_list[idx][0].tolist()],
            "val": dataset[dataset.idx_list[idx][1].tolist()],
        }
        for idx in range(n_splits)
    ]
    graph_set_fold_id(dataset, 0)

    return dataset


def graph_set_fold_id(dataset, fold_id):
    r"""Set the current fold id of graph dataset.

    Parameters
    ----------
    dataset: ``torch_geometric.data.dataset.Dataset``
        dataset with multiple graphs.

    fold_id: ``int``
        The current fold id this dataset uses. Should be in [0, dataset.n_splits)

    Returns
    -------
    ``torch_geometric.data.dataset.Dataset``
        The reference original dataset.
    """
    if not hasattr(dataset, "n_splits"):
        raise ValueError("Dataset set fold id before cross validated!")
    assert (
        0 <= fold_id < dataset.n_splits
    ), "Fold id %d exceed total cross validation split number %d" % (
        fold_id,
        dataset.n_splits,
    )
    dataset.current_fold_id = fold_id
    dataset.train_split = dataset.cv_dict[dataset.current_fold_id]["train"]
    dataset.val_split = dataset.cv_dict[dataset.current_fold_id]["val"]
    dataset.train_index = dataset.idx_list[dataset.current_fold_id][0]
    dataset.val_index = dataset.idx_list[dataset.current_fold_id][1]
    return dataset


def graph_random_splits(dataset, train_ratio=0.2, val_ratio=0.4, seed=None):
    r"""Splitting graph dataset with specific ratio for train/val/test.

    Parameters
    ----------
    dataset: ``torch_geometric.data.dataset.Dataset``
        dataset with multiple graphs.

    train_ratio : float
        the portion of data that used for training.

    val_ratio : float
        the portion of data that used for validation.

    seed : int
        random seed for splitting dataset.

    Returns
    -------
    ``torch_geometric.data.dataset.Dataset``
        The reference of original dataset
    """

    assert (
        train_ratio + val_ratio <= 1
    ), "the sum of train_ratio and val_ratio is larger than 1"
    r_s = torch.get_rng_state()
    if torch.cuda.is_available():
        r_s_cuda = torch.cuda.get_rng_state()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    perm = torch.randperm(len(dataset))
    train_index = perm[: int(len(dataset) * train_ratio)]
    val_index = perm[
        int(len(dataset) * train_ratio) : int(len(dataset) * (train_ratio + val_ratio))
    ]
    test_index = perm[int(len(dataset) * (train_ratio + val_ratio)) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    # set train_idx, val_idx and test_idx as dataset attribute
    dataset.train_split = train_dataset
    dataset.val_split = val_dataset
    dataset.test_split = test_dataset

    dataset.train_index = train_index
    dataset.val_index = val_index
    dataset.test_index = test_index

    torch.set_rng_state(r_s)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(r_s_cuda)

    return dataset


def graph_get_split(
    dataset, mask="train", is_loader=True, batch_size=128, num_workers=0
):
    r"""Get train/test dataset/dataloader after cross validation.

    Parameters
    ----------
    dataset: ``torch_geometric.data.dataset.Dataset``
        dataset with multiple graphs.

    mask : str
        return with which dataset/dataloader

    is_loader : bool
        return with autogl.datasets or pyg.Dataloader

    batch_size : int
        batch_size for generateing Dataloader

    """
    assert hasattr(
        dataset, "%s_split" % (mask)
    ), "Given dataset do not have %s split" % (mask)
    if is_loader:
        return DataLoader(
            getattr(dataset, "%s_split" % (mask)),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        return getattr(dataset, "%s_split" % (mask))


'''
def graph_cross_validation(dataset, n_splits = 10, shuffle = True, random_seed = 42, fold_idx = 0, batch_size = 32, dataloader = True):
    r"""Cross validation for graph classification data, returning one fold with specific idx in autogl.datasets or pyg.Dataloader(default)

    Parameters
    ----------
    dataset : str
        dataset with multiple graphs.

    n_splits : int
        the number of how many folds will be splitted.

    shuffle : bool
        shuffle or not for sklearn.model_selection.StratifiedKFold

    random_seed : int
        random_state for sklearn.model_selection.StratifiedKFold

    fold_idx : int
        specific fold id from 0 to n_splits-1

    batch_size : int
        batch_size for generateing Dataloader

    dataloader : bool
        return with autogl.datasets or pyg.Dataloader
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle = shuffle, random_state = random_seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(dataset.data.y)), dataset.data.y):
        idx_list.append(idx)
    assert 0 <= fold_idx and fold_idx < n_splits, "fold_idx must be from 0 to " + str(n_splits-1)
    train_idx, test_idx = idx_list[fold_idx]
    test_dataset = dataset[test_idx.tolist()]
    train_dataset = dataset[train_idx.tolist()]
    if dataloader:
        return DataLoader(train_dataset, batch_size=128), DataLoader(test_dataset, batch_size=128)
    else:
        return train_dataset, test_dataset
'''


def train_test_split(self, method="auto", ratio=None):
    raise NotImplementedError()


def train_valid_split(self, method="auto", ratio=None):
    raise NotImplementedError()


def cross_validation_split(self, method="auto", cv_fold_num=5):
    return NotImplementedError()


# below get_* can also be set as property
def get_train_dataset(self):
    raise NotImplementedError()


def get_test_dataset(self):
    raise NotImplementedError()


def get_valid_dataset(self):
    raise NotImplementedError()


def get_train_generator(self, batch_size):
    """
    should return a torch.utils.data.Dataloader
    """
    raise NotImplementedError()


def get_test_generator(self, batch_size):
    """
    should return a torch.utils.data.Dataloader
    """
    raise NotImplementedError()


def get_valid_generator(self, batch_size):
    """
    should return a torch.utils.data.Dataloader
    """
    raise NotImplementedError()
