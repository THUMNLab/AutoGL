import numpy as np
import torch


def data_is_tensor(data):
    return isinstance(data.x, torch.Tensor)


def data_is_numpy(data):
    return isinstance(data.x, np.ndarray)


def data_tensor2np(data):
    if data_is_tensor(data):
        data.x = data.x.numpy()
        data.y = data.y.numpy()
        data.edge_index = data.edge_index.numpy()
    return data


def data_np2tensor(data):
    if not data_is_tensor(data):
        if data_is_numpy(data):
            data.x = torch.FloatTensor(data.x)
        data.y = torch.tensor(data.y)
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    return data


# from .base import BaseFeatureAtom
# class DataTensor2Np(BaseFeatureAtom):
#     def __call__(self,data):
#         return data_tensor2np(data)
# class DataNp2Tensor(BaseFeatureAtom):
#     def __call__(self,data):
#         return data_np2tensor(data)
