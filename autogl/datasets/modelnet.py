# import os.path as osp
# import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from . import register_dataset


class ModelNet10(ModelNet):
    def __init__(self, path: str, train: bool):
        # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ModelNet10, self).__init__(path, name="10", train=train)


class ModelNet40(ModelNet):
    def __init__(self, path: str, train: bool):
        # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(ModelNet40, self).__init__(path, name="40", train=train)


@register_dataset("ModelNet10Train")
class ModelNet10Train(ModelNet):
    def __init__(self, path: str):
        super(ModelNet10Train, self).__init__(path, "10", train=True)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ModelNet10Train, self).get(idx)


@register_dataset("ModelNet10Test")
class ModelNet10Test(ModelNet):
    def __init__(self, path: str):
        super(ModelNet10Test, self).__init__(path, "10", train=False)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ModelNet10Test, self).get(idx)


@register_dataset("ModelNet40Train")
class ModelNet40Train(ModelNet):
    def __init__(self, path: str):
        super(ModelNet40Train, self).__init__(path, "40", train=True)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ModelNet40Train, self).get(idx)


@register_dataset("ModelNet40Test")
class ModelNet40Test(ModelNet):
    def __init__(self, path: str):
        super(ModelNet40Test, self).__init__(path, "40", train=False)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ModelNet40Test, self).get(idx)
