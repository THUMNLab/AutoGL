"""
auto graph model
a list of models with their hyper parameters
NOTE: neural architecture search (NAS) maybe included here
"""
import copy
import logging
import typing as _typing
import torch
import torch.nn.functional as F
from copy import deepcopy

base_approach_logger: logging.Logger = logging.getLogger("BaseModel")


def activate_func(x, func):
    if func == "tanh":
        return torch.tanh(x)
    elif hasattr(F, func):
        return getattr(F, func)(x)
    elif func == "":
        pass
    else:
        raise TypeError("PyTorch does not support activation function {}".format(func))

    return x


class BaseModel:
    def __init__(self, init=False, *args, **kwargs):
        super(BaseModel, self).__init__()

    def get_hyper_parameter(self):
        return deepcopy(self.hyperparams)

    @property
    def hyper_parameter_space(self):
        return self.space

    @hyper_parameter_space.setter
    def hyper_parameter_space(self, space):
        self.space = space

    def initialize(self):
        pass

    def forward(self):
        pass

    def to(self, device):
        if isinstance(device, (str, torch.device)):
            self.device = device
        if (
            hasattr(self, "model")
            and self.model is not None
            and isinstance(self.model, torch.nn.Module)
        ):
            self.model.to(self.device)
        return self

    def from_hyper_parameter(self, hp):
        ret_self = self.__class__(
            num_features=self.num_features,
            num_classes=self.num_classes,
            device=self.device,
            init=False,
        )
        ret_self.hyperparams.update(hp)
        ret_self.params.update(self.params)
        ret_self.initialize()
        return ret_self

    def get_num_classes(self):
        return self.num_classes

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.params["num_class"] = num_classes

    def get_num_features(self):
        return self.num_features

    def set_num_features(self, num_features):
        self.num_features = num_features
        self.params["features_num"] = self.num_features

    def set_num_graph_features(self, num_graph_features):
        assert hasattr(
            self, "num_graph_features"
        ), "Cannot set graph features for tasks other than graph classification"
        self.num_graph_features = num_graph_features
        self.params["num_graph_features"] = num_graph_features


class _BaseBaseModel:
    # todo: after renaming the experimental base class _BaseModel to BaseModel,
    #       rename this class to _BaseModel
    """
    The base class for class BaseModel,
    designed to implement some basic functionality of BaseModel.
    --  Designed by ZiXin Sun
    """

    @classmethod
    def __formulate_device(
        cls, device: _typing.Union[str, torch.device] = ...
    ) -> torch.device:
        if type(device) == torch.device or (
            type(device) == str and device.strip().lower() != "auto"
        ):
            return torch.device(device)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, __device: _typing.Union[str, torch.device, None]):
        self.__device: torch.device = self.__formulate_device(__device)

    @property
    def model(self) -> _typing.Optional[torch.nn.Module]:
        if self._model is None:
            base_approach_logger.debug(
                "property of model NOT initialized before accessing"
            )
        return self._model

    @model.setter
    def model(self, _model: torch.nn.Module) -> None:
        if not isinstance(_model, torch.nn.Module):
            raise TypeError(
                "the property of model MUST be an instance of " "torch.nn.Module"
            )
        self._model = _model

    def _initialize(self):
        raise NotImplementedError

    def initialize(self) -> bool:
        """
        Initialize the model in case that the model has NOT been initialized
        :return: whether self._initialize() method called
        """
        if not self.__is_initialized:
            self._initialize()
            self.__is_initialized = True
            return True
        return False

    # def to(self, *args, **kwargs):
    #     """
    #     Due to the signature of to() method in class BaseApproach
    #     is inconsistent with the signature of the method
    #     in the base class torch.nn.Module,
    #     this intermediate overridden method is necessary to
    #     walk around (bypass) the inspection for
    #     signature of overriding method.
    #     :param args: positional arguments list
    #     :param kwargs: keyword arguments dict
    #     :return: self
    #     """
    #     return super(_BaseBaseModel, self).to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            return self.model(*args, **kwargs)
        else:
            raise NotImplementedError

    def __init__(
        self,
        model: _typing.Optional[torch.nn.Module] = None,
        initialize: bool = False,
        device: _typing.Union[str, torch.device] = ...,
    ):
        if type(initialize) != bool:
            raise TypeError
        super(_BaseBaseModel, self).__init__()
        self.__device: torch.device = self.__formulate_device(device)
        self._model: _typing.Optional[torch.nn.Module] = model
        self.__is_initialized: bool = False
        if initialize:
            self.initialize()


class _BaseModel(_BaseBaseModel, BaseModel):
    """
    The upcoming root base class for Model, i.e. BaseModel
    --  Designed by ZiXin Sun
    """

    # todo: Deprecate and remove the legacy class "BaseModel",
    #       then rename this class to "BaseModel",
    #       correspondingly, this class will no longer extend
    #       the legacy class "BaseModel" after the removal.
    def _initialize(self):
        raise NotImplementedError

    def to(self, device: torch.device):
        self.device = device
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
        return super().to(device)

    @property
    def space(self) -> _typing.Sequence[_typing.Dict[str, _typing.Any]]:
        # todo: deprecate and remove in future major version
        return self.__hyper_parameter_space

    @property
    def hyper_parameter_space(self):
        return self.__hyper_parameter_space

    @hyper_parameter_space.setter
    def hyper_parameter_space(
        self, space: _typing.Sequence[_typing.Dict[str, _typing.Any]]
    ):
        self.__hyper_parameter_space = space

    @property
    def hyper_parameter(self) -> _typing.Dict[str, _typing.Any]:
        return self.__hyper_parameter

    @hyper_parameter.setter
    def hyper_parameter(self, _hyper_parameter: _typing.Dict[str, _typing.Any]):
        if not isinstance(_hyper_parameter, dict):
            raise TypeError
        self.__hyper_parameter = _hyper_parameter

    def get_hyper_parameter(self) -> _typing.Dict[str, _typing.Any]:
        """
        todo: consider deprecating this trivial getter method in the future
        :return: copied hyper parameter
        """
        return copy.deepcopy(self.__hyper_parameter)

    def __init__(
        self,
        model: _typing.Optional[torch.nn.Module] = None,
        initialize: bool = False,
        hyper_parameter_space: _typing.Sequence[_typing.Any] = ...,
        hyper_parameter: _typing.Dict[str, _typing.Any] = ...,
        device: _typing.Union[str, torch.device] = ...,
    ):
        if type(initialize) != bool:
            raise TypeError
        super(_BaseModel, self).__init__(model, initialize, device)
        if hyper_parameter_space != Ellipsis and isinstance(
            hyper_parameter_space, _typing.Sequence
        ):
            self.__hyper_parameter_space: _typing.Sequence[
                _typing.Dict[str, _typing.Any]
            ] = hyper_parameter_space
        else:
            self.__hyper_parameter_space: _typing.Sequence[
                _typing.Dict[str, _typing.Any]
            ] = []
        if hyper_parameter != Ellipsis and isinstance(hyper_parameter, dict):
            self.__hyper_parameter: _typing.Dict[str, _typing.Any] = hyper_parameter
        else:
            self.__hyper_parameter: _typing.Dict[str, _typing.Any] = {}

    def from_hyper_parameter(self, hyper_parameter: _typing.Dict[str, _typing.Any]):
        raise NotImplementedError


class ClassificationModel(_BaseModel):
    def _initialize(self):
        raise NotImplementedError

    def from_hyper_parameter(
        self, hyper_parameter: _typing.Dict[str, _typing.Any]
    ) -> "ClassificationModel":
        new_model: ClassificationModel = self.__class__(
            num_features=self.num_features,
            num_classes=self.num_classes,
            device=self.device,
            init=False,
        )
        _hyper_parameter = self.hyper_parameter
        _hyper_parameter.update(hyper_parameter)
        new_model.hyper_parameter = _hyper_parameter
        new_model.initialize()
        return new_model

    def __init__(
        self,
        num_features: int = ...,
        num_classes: int = ...,
        num_graph_features: int = ...,
        device: _typing.Union[str, torch.device] = ...,
        hyper_parameter_space: _typing.Sequence[_typing.Any] = ...,
        hyper_parameter: _typing.Dict[str, _typing.Any] = ...,
        init: bool = False,
        **kwargs
    ):
        if "initialize" in kwargs:
            del kwargs["initialize"]
        super(ClassificationModel, self).__init__(
            initialize=init,
            hyper_parameter_space=hyper_parameter_space,
            hyper_parameter=hyper_parameter,
            device=device,
            **kwargs
        )
        if num_classes != Ellipsis and type(num_classes) == int:
            self.__num_classes: int = num_classes if num_classes > 0 else 0
        else:
            self.__num_classes: int = 0
        if num_features != Ellipsis and type(num_features) == int:
            self.__num_features: int = num_features if num_features > 0 else 0
        else:
            self.__num_features: int = 0
        if num_graph_features != Ellipsis and type(num_graph_features) == int:
            if num_graph_features > 0:
                self.__num_graph_features: int = num_graph_features
            else:
                self.__num_graph_features: int = 0
        else:
            self.__num_graph_features: int = 0

    def __repr__(self) -> str:
        import yaml

        return yaml.dump(self.hyper_parameter)

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, __num_classes: int):
        if type(__num_classes) != int:
            raise TypeError
        if not __num_classes > 0:
            raise ValueError
        self.__num_classes = __num_classes if __num_classes > 0 else 0

    @property
    def num_features(self) -> int:
        return self.__num_features

    @num_features.setter
    def num_features(self, __num_features: int):
        if type(__num_features) != int:
            raise TypeError
        if not __num_features > 0:
            raise ValueError
        self.__num_features = __num_features if __num_features > 0 else 0

    def get_num_classes(self) -> int:
        # todo: consider replacing with property with getter and setter
        return self.__num_classes

    def set_num_classes(self, num_classes: int) -> None:
        # todo: consider replacing with property with getter and setter
        if type(num_classes) != int:
            raise TypeError
        self.__num_classes = num_classes if num_classes > 0 else 0

    def get_num_features(self) -> int:
        # todo: consider replacing with property with getter and setter
        return self.__num_features

    def set_num_features(self, num_features: int):
        # todo: consider replacing with property with getter and setter
        if type(num_features) != int:
            raise TypeError
        self.__num_features = num_features if num_features > 0 else 0

    def set_num_graph_features(self, num_graph_features: int):
        # todo: consider replacing with property with getter and setter
        if type(num_graph_features) != int:
            raise TypeError
        else:
            if num_graph_features > 0:
                self.__num_graph_features = num_graph_features
            else:
                self.__num_graph_features = 0


class _ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(_ClassificationModel, self).__init__()

    def cls_encode(self, data) -> torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cls_forward(self, data) -> torch.Tensor:
        return self.cls_decode(self.cls_encode(data))


class ClassificationSupportedSequentialModel(_ClassificationModel):
    def __init__(self):
        super(ClassificationSupportedSequentialModel, self).__init__()

    @property
    def sequential_encoding_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError

    def cls_encode(self, data) -> torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
