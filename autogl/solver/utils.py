"""
Utilities used by the solver

* LeaderBoard: The LeaderBoard that maintains the performance of models.
"""

import random
import typing as _typing
import torch
import torch.backends.cudnn
import numpy as np
import pandas as pd
from ..backend import DependentBackend
from ..data import Dataset
from ..data.graph import GeneralStaticGraph

from ..utils import get_logger
LOGGER = get_logger("LeaderBoard")

BACKEND = DependentBackend.get_backend_name()

if BACKEND == 'dgl':
    from autogl.datasets.utils.conversion import to_dgl_dataset as _convert_dataset
else:
    from autogl.datasets.utils.conversion import to_pyg_dataset as _convert_dataset

class LeaderBoard:
    """
    The leaderBoard that can be used to store / sort the model performance automatically.

    Parameters
    ----------
    fields: list of `str`
        A list of field name that shows the model performance. The first field is used as
        the major field for sorting the model performances.

    is_higher_better: `dict` of *field* -> `bool`
        A mapping of indicator that whether each field is higher better.
    """

    def __init__(self, fields, is_higher_better):
        assert isinstance(fields, list)
        self.keys = ["name"] + fields
        self.perform_dict = pd.DataFrame(columns=self.keys)
        self.is_higher_better = is_higher_better
        self.major_field = fields[0]

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
        if field in self.keys and not field == "name":
            self.major_field = field
        else:
            LOGGER.warning(
                f"Field [{field}] NOT found in the current LeaderBoard, will ignore."
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
        top_k: int = top_k if top_k > 0 else len(self.perform_dict)

        """
        reindex self.__performance_data_frame
        to ensure the columns of name and representation are in left-side of the data frame
        """
        _columns = self.perform_dict.columns.tolist()
        maxcolwidths: _typing.List[_typing.Optional[int]] = []
        if "name" in _columns:
            _columns.remove("name")
            _columns.insert(0, "name")
            maxcolwidths.append(40)
        self.perform_dict = self.perform_dict[_columns]

        sorted_performance_df: pd.DataFrame = self.perform_dict.sort_values(
            self.major_field, ascending=not self.is_higher_better[self.major_field]
        )
        sorted_performance_df = sorted_performance_df.head(top_k)

        from tabulate import tabulate

        _columns = sorted_performance_df.columns.tolist()
        maxcolwidths.extend([None for _ in range(len(_columns) - len(maxcolwidths))])
        print(
            tabulate(
                list(zip(*[sorted_performance_df[column] for column in _columns])),
                headers=_columns,
                tablefmt="grid",
            )
        )

def get_graph_from_dataset(dataset, graph_id=0):
    if isinstance(dataset, Dataset):
        return dataset[graph_id]
    if BACKEND == 'pyg': return dataset[graph_id]
    if BACKEND == 'dgl':
        from dgl import DGLGraph
        data = dataset[graph_id]
        if isinstance(data, DGLGraph): return data
        return data[0]
    
def get_graph_node_number(graph):
    # FIXME: if the feature is None, this will throw an error
    if isinstance(graph, GeneralStaticGraph):
        if BACKEND == 'pyg':
            return graph.nodes.data['x'].size(0)
        return graph.nodes.data['feat'].size(0)
    if BACKEND == 'pyg':
        size = graph.x.shape[0]
    else:
        size = graph.num_nodes()
    return size

def get_graph_node_features(graph):
    if isinstance(graph, GeneralStaticGraph):
        if BACKEND == 'dgl' and 'feat' in graph.nodes.data:
            return graph.nodes.data['feat']
        if BACKEND == 'pyg' and 'x' in graph.nodes.data:
            return graph.nodes.data['x']
        return None
    if BACKEND == 'pyg' and hasattr(graph, 'x'):
        return graph.x
    elif BACKEND == 'dgl' and 'feat' in graph.ndata:
        return graph.ndata['feat']
    return None

def get_graph_masks(graph, mask='train'):
    if isinstance(graph, GeneralStaticGraph):
        if f'{mask}_mask' in graph.nodes.data:
            return graph.nodes.data[f'{mask}_mask']
        return None
    if BACKEND == 'pyg' and hasattr(graph, f'{mask}_mask'):
        return getattr(graph, f'{mask}_mask')
    if BACKEND == 'dgl' and f'{mask}_mask' in graph.ndata:
        return graph.ndata[f'{mask}_mask']
    return None

def get_graph_labels(graph):
    if isinstance(graph, GeneralStaticGraph):
        if 'label' in graph.nodes.data and BACKEND == 'dgl':
            return graph.nodes.data['label']
        if 'y' in graph.nodes.data and BACKEND == 'pyg':
            return graph.nodes.data['y']
        return None
    if BACKEND == 'pyg' and hasattr(graph, 'y'): return graph.y
    if BACKEND == 'dgl' and 'label' in graph.ndata: return graph.ndata['label']
    return None

def get_dataset_labels(dataset):
    if isinstance(dataset[0], GeneralStaticGraph):
        return torch.LongTensor([d.data['label' if BACKEND == 'dgl' else 'y'] for d in dataset])
    if BACKEND == 'pyg':
        return dataset.data.y
    else:
        return torch.LongTensor([d[1] for d in dataset])

def convert_dataset(dataset):
    # todo: replace the trick by re-implementing the convert_dataset in utils
    if hasattr(dataset[0], "edges"): return _convert_dataset(dataset)
    # if isinstance(dataset, Dataset): return _convert_dataset(dataset)
    return dataset

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

def get_graph_labels_hetero(graph, target_node_type):
    if isinstance(graph, GeneralStaticGraph):
        if 'label' in graph.nodes[target_node_type].data and BACKEND == 'dgl':
            return graph.nodes[target_node_type].data['label']
        return None
    if BACKEND == 'dgl' and 'label' in graph.ndata[target_node_type]: return graph.ndata[target_node_type]['label']
