# an auto class that supports basic space and register settings
# TODO: consider moving this module to hpo or other places

from typing import Iterable

class AutoModule(object):
    # this module is only responsible for single level
    def __init__(self):
        # used to store hp space
        self.__hyper_parameter_space = []

        # used to store current hp
        self.__hyper_parameter = {}
    
    @property
    def hyper_parameter_space(self):
        return self.__hyper_parameter_space
    
    @hyper_parameter_space.setter
    def hyper_parameter_space(self, hp_space: Iterable):
        self.__hyper_parameter_space = []
        for hps in hp_space:
            self.register_hyper_parameter_space(hps)

    @property
    def hyper_parameter(self):
        return self.__hyper_parameter
    
    @hyper_parameter.setter
    def hyper_parameter(self, hp: dict):
        for key, value in hp.items():
            self.register_hyper_parameter(key, value, False)

    def register_hyper_parameter_space(self, new_hp_space: dict):
        self.__hyper_parameter_space.append(new_hp_space)
    
    def register_hyper_parameter(self, key, value, _check=True):
        self.__hyper_parameter[key] = value
        if _check:
            assert not hasattr(self, key), "Replicate key {}!".format(key)
        setattr(self, key, value)
