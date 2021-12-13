# import torch
# import typing as _typing
# from autogl.utils.autobase import AutoModule
#
# class BaseEncoder(torch.nn.Module):
#     def forward(self, data):
#         raise NotImplementedError()
#
# class BaseAutoEncoder(AutoModule):
#     def __init__(self, device: _typing.Union[str, torch.device]="auto"):
#         super().__init__()
#         self.device = device
#         self.model = None
#
#     def initialize(self): raise NotImplementedError
#     def from_hyper_parameter(self, hp): raise NotImplementedError
#
#     @property
#     def device(self):
#         return self.__device
#
#     @device.setter
#     def device(self, dev):
#         if dev == "auto":
#             dev = "cpu" if not torch.cuda.is_available() else "cuda"
#         self.__device = torch.device(dev)
#
#     @property
#     def model(self):
#         return self.__model
#
#     @model.setter
#     def model(self, m: BaseEncoder):
#         self.__model = m
#
#     def to(self, device):
#         self.device = device
#         if self.model is not None:
#             self.model.to(device)
#
# class AutoHomogeneousEncoder(BaseAutoEncoder):
#     def __init__(self, num_features, device: _typing.Union[str, torch.device] = "auto"):
#         super().__init__(device=device)
#         self.num_features = num_features
#
#     @property
#     def num_features(self):
#         return self.__num_features
#
#     @num_features.setter
#     def num_features(self, num_features):
#         self.__num_features = num_features
