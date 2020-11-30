"""
Util tools used by solver

* leaderboard: The leaderboard that maintains the performance of models.
"""

import random

import torch
import numpy as np
import pandas as pd

from ..utils import get_logger

LOGGER = get_logger("leaderboard")


class Leaderboard:
    """
    The leaderboard that can be used to store / sort the model performance automatically.

    Parameters
    ----------
    fields: list of `str`
        A list of field name that shows the model performance. The first field is used as
        the major field for sorting the model performances.

    is_higher_better: list of `bool`
        A list of indicator that whether the field score is higher better.
    """

    def __init__(self, fields, is_higher_better):
        assert isinstance(fields, list)
        self.keys = ["name"] + fields
        self.perform_dict = pd.DataFrame(columns=self.keys)
        self.is_higher_better = is_higher_better
        self.major_field = fields[0]

    def set_major_field(self, field) -> None:
        """
        Set the major field of current leaderboard.

        Parameters
        ----------
        field: `str`
            The major field, should be one of the fields when initialized.

        Returns
        -------
        None
        """
        if field in self.keys and not field == "name":
            self.major_field = field
        else:
            LOGGER.warning(
                "do not find major field %s in current leaderboard, will ignore.", field
            )

    def insert_model_performance(self, name, performance) -> None:
        """
        Add/Override a record of model performance. If name given is already in the leaderboard,
        will overrride the slot.

        Parameters
        ----------
        name: `str`
            The model name/identifier that identifies the model.

        performance: `dict`
            The performance dict. The key inside the dict should be the fields when initialized.
            The value of the dict should be the corresponding scores.

        Returns
        -------
        None
        """
        if name not in self.perform_dict["name"]:
            # we just add a new row
            performance["name"] = name
            new = pd.DataFrame(performance, index=[0])
            self.perform_dict = self.perform_dict.append(new, ignore_index=True)
        else:
            LOGGER.warning(
                "model already in the leaderboard, will override current result."
            )
            self.remove_model_performance(name)
            self.insert_model_performance(name, performance)

    def remove_model_performance(self, name) -> None:
        """
        Remove the record of given models.

        Parameters
        ----------
        name: `str`
            The model name/identifier that needed to be removed.

        Returns
        -------
        None
        """
        if name not in self.perform_dict["name"]:
            LOGGER.warning(
                "no model detected in current leaderboard, will ignore removing action."
            )
            return
        index = self.perform_dict["name"][self.perform_dict["name"] == name].index
        self.perform_dict.drop(self.perform_dict.index[index], inplace=True)
        return

    def get_best_model(self, index=0) -> str:
        """
        Get the best model according to the performance of the major field.

        Parameters
        ----------
        index: `int`
            The index of the model (from good to bad). Default `0`.

        Returns
        -------
        name: `str`
            The name/identifier of the required model.
        """
        sorted_df = self.perform_dict.sort_values(
            by=self.major_field, ascending=not self.is_higher_better[self.major_field]
        )
        name_list = sorted_df["name"].tolist()
        if "ensemble" in name_list:
            name_list.remove("ensemble")
        return name_list[index]

    def show(self, top_k=-1) -> None:
        """
        Show current leaderboard (from good model to bad).

        Parameters
        ----------
        top_k: `int`
            Controls the number model shown. If below `0`, will show all the models. Default `-1`.

        Returns
        -------
        None
        """
        if top_k == -1:
            top_k = len(self.perform_dict["name"])
        print(
            self.perform_dict.sort_values(
                by=self.major_field,
                ascending=not self.is_higher_better[self.major_field],
            ).head(top_k)
        )


def set_seed(seed=None):
    """
    Set seed of whole process
    """
    if seed is None:
        seed = random.randint(0, 5000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
