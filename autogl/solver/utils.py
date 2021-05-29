"""
Utilities used by the solver

* LeaderBoard: The LeaderBoard that maintains the performance of models.
"""

import random
import typing as _typing
import torch.backends.cudnn
import numpy as np
import pandas as pd

from ..utils import get_logger

LOGGER = get_logger("LeaderBoard")


class LeaderBoard:
    """
    The LeaderBoard that can be used to store / sort the model performance automatically.

    Parameters
    ----------
    fields: list of `str`
        A list of field name that shows the model performance. The first field is used as
        the major field for sorting the model performances.

    is_higher_better: list of `bool`
        A list of indicator that whether the field score is higher better.
    """

    def __init__(
            self, fields: _typing.Sequence[str],
            is_higher_better: _typing.Union[
                _typing.Sequence[bool],
                _typing.Dict[str, bool]
            ]
    ):
        if not isinstance(fields, _typing.Sequence):
            raise TypeError
        for _field in fields:
            if type(_field) != str:
                raise TypeError
        if isinstance(is_higher_better, dict):
            self.__is_higher_better: _typing.Sequence[bool] = [
                bool(is_higher_better[field]) for field in fields
            ]
        elif isinstance(is_higher_better, _typing.Sequence):
            self.__is_higher_better: _typing.Sequence[bool] = [
                bool(item) for item in is_higher_better
            ]
        else:
            raise TypeError
        self.__fields: _typing.Sequence[str] = fields
        self.__major_field: str = fields[0]

        self.__performance_data_frame: pd.DataFrame = pd.DataFrame(
            columns=["name", "representation"] + list(fields)
        )

    def set_major_field(self, field) -> None:
        """
        Set the major field of current LeaderBoard.

        Parameters
        ----------
        field: `str`
            The major field, should be one of the fields when initialized.

        Returns
        -------
        None
        """
        if field in self.__fields:
            self.__major_field = field
        else:
            LOGGER.warning(
                "do not find major field %s in the current LeaderBoard, will ignore.", field
            )

    def add_performance(
            self, name: str,
            representation: _typing.Union[str, _typing.Dict[str, _typing.Any]],
            performance: _typing.Dict[str, float]
    ) -> 'LeaderBoard':
        """
        Add a record of model performance.

        Parameters
        ----------
        name: `str`
            The model name/identifier that identifies the model.

        representation: `str` or `dict`
            The representation of the corresponding methodology.

        performance: `dict`
            The performance dict. The key inside the dict should be the fields when initialized.
            The value of the dict should be the corresponding scores.

        Returns
        -------
        self:
            this `LeaderBoard` instance for chained call
        """
        import yaml
        if isinstance(representation, dict):
            __representation: str = yaml.dump(representation)
        elif isinstance(representation, str):
            __representation: str = representation
        else:
            raise TypeError

        __dict = {"name": name, "representation": __representation}
        __dict.update(performance)
        self.__performance_data_frame = self.__performance_data_frame.append(
            pd.DataFrame(__dict, index=[0]), ignore_index=True
        )
        return self

    def insert_model_performance(
            self, name: str, performance: _typing.Dict[str, _typing.Any]
    ) -> None:
        """
        Add a record of model performance.
        todo: This method will be deprecated

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
        self.add_performance(name, name, performance)

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
        sorted_performance_df = self.__performance_data_frame.sort_values(
            self.__major_field,
            ascending=not (
                dict(zip(self.__fields, self.__is_higher_better))[self.__major_field]
                if self.__major_field in self.__fields else True
            )
        )
        name_list = sorted_performance_df["name"].tolist()
        if "ensemble" in name_list:
            name_list.remove("ensemble")
        return name_list[index]

    def show(self, top_k=0) -> None:
        """
        Show current LeaderBoard (from best model to worst).

        Parameters
        ----------
        top_k: `int`
            Controls the number model shown.
            If less than or equal to `0`, will show all the models. Default to `0`.

        Returns
        -------
        None
        """
        top_k: int = top_k if top_k > 0 else len(self.__performance_data_frame)

        '''
        reindex self.__performance_data_frame
        to ensure the columns of name and representation are in left-side of the data frame
        '''
        _columns = self.__performance_data_frame.columns.tolist()
        maxcolwidths: _typing.List[_typing.Optional[int]] = []
        if "representation" in _columns:
            _columns.remove("representation")
            _columns.insert(0, "representation")
            maxcolwidths.append(40)
        if "name" in _columns:
            _columns.remove("name")
            _columns.insert(0, "name")
            maxcolwidths.append(40)
        self.__performance_data_frame = self.__performance_data_frame[_columns]

        sorted_performance_df: pd.DataFrame = self.__performance_data_frame.sort_values(
            self.__major_field,
            ascending=not (
                dict(zip(self.__fields, self.__is_higher_better))[self.__major_field]
                if self.__major_field in self.__fields else True
            )
        )
        sorted_performance_df = sorted_performance_df.head(top_k)

        from tabulate import tabulate
        _columns = sorted_performance_df.columns.tolist()
        maxcolwidths.extend([None for _ in range(len(_columns) - len(maxcolwidths))])
        print(
            tabulate(
                list(zip(*[sorted_performance_df[column] for column in _columns])),
                headers=_columns, tablefmt="grid"
            )
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
