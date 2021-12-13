# import torch
# import typing as _typing
# from autogl.utils.autobase import AutoModule
#
# class BaseDecoder(torch.nn.Module):
#     def forward(self, features: _typing.Iterable[torch.Tensor], data):
#         raise NotImplementedError()
#
# class BaseAutoDecoder(AutoModule):
#     def __init__(self, device: _typing.Union[str, torch.device]="auto"):
#         super().__init__()
#         self.device = device
#         self.model = None
#
#     def initialize(self, encoder):
#         raise NotImplementedError
#
#     def from_hyper_parameter_and_encoder(self, hp, encoder):
#         raise NotImplementedError
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
#     def model(self, m: BaseDecoder):
#         self.__model = m
#
#     def to(self, device):
#         self.device = device
#         if self.model is not None:
#             self.model.to(device)
#
#
# class AutoClassifierDecoder(BaseAutoDecoder):
#     def __init__(self, num_classes=None, device: _typing.Union[str, torch.device] = "auto"):
#         super().__init__(device=device)
#         self.num_classes = num_classes
#
#     @property
#     def num_classes(self):
#         return self.__num_classes
#
#     @num_classes.setter
#     def num_classes(self, num_classes):
#         self.__num_classes = num_classes
#
