from copy import deepcopy
from ..base import BaseAutoModel

class BaseHeteroModelMaintainer(BaseAutoModel):
    def __init__(self, num_features, num_classes, device, dataset=None, **kwargs):
        super().__init__(num_features, num_classes, device, **kwargs)
        self._registered_parameters = {}
        if dataset is not None:
            self.from_dataset(dataset)

    def from_dataset(self, dataset):
        raise NotImplementedError

    # consider moving this to inner classes
    def register_parameter(self, key: str, value):
        self._registered_parameters[key] = value
        setattr(self, key, value)

    def destroy_parameter(self, key):
        if key in self._registered_parameters:
            return self._registered_parameters.pop(key)
        return None

    def from_hyper_parameter(self, hp, **kwargs):
        kw = deepcopy(self._kwargs)
        kw.update(kwargs)
        ret_self = self.__class__(
            self.input_dimension,
            self.output_dimension,
            self.device,
            **kw
        )
        hp_now = dict(self.hyper_parameters)
        hp_now.update(hp)
        ret_self.hyper_parameters = hp_now
        for key, value in self._registered_parameters.items():
            ret_self.register_parameter(key, value)
        ret_self.initialize()
        return ret_self
