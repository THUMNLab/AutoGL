import os.path as osp
import os
import torch
from ..data.dataset import Dataset


try:
    import torch_geometric
except ImportError:
    torch_geometric = None
    pyg = False
else:
    pyg = True

DATASET_DICT = {}


def register_dataset(name):
    """
    New dataset types can be added to autogl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_DICT:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, Dataset) and (
            pyg and not issubclass(cls, torch_geometric.data.Dataset)
        ):
            raise ValueError(
                "Dataset ({}: {}) must extend autogl.data.Dataset".format(
                    name, cls.__name__
                )
            )
        DATASET_DICT[name] = cls
        return cls

    return register_dataset_cls


from .pyg import (
    AmazonComputersDataset,
    AmazonPhotoDataset,
    CoauthorPhysicsDataset,
    CoauthorCSDataset,
    CoraDataset,
    CiteSeerDataset,
    PubMedDataset,
    RedditDataset,
    MUTAGDataset,
    IMDBBinaryDataset,
    IMDBMultiDataset,
    CollabDataset,
    ProteinsDataset,
    REDDITBinary,
    REDDITMulti5K,
    REDDITMulti12K,
    PTCMRDataset,
    NCI1Dataset,
    ENZYMES,
    QM9Dataset,
)
from .ogb import (
    OGBNproductsDataset,
    OGBNproteinsDataset,
    OGBNarxivDataset,
    OGBNpapers100MDataset,
    OGBNmagDataset,
    OGBGmolhivDataset,
    OGBGmolpcbaDataset,
    OGBGppaDataset,
    OGBGcodeDataset,
    OGBLppaDataset,
    OGBLcollabDataset,
    OGBLddiDataset,
    OGBLcitationDataset,
    OGBLwikikgDataset,
    OGBLbiokgDataset,
)
from .gatne import GatneDataset, AmazonDataset, TwitterDataset, YouTubeDataset
from .gtn_data import GTNDataset, ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from .han_data import HANDataset, ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from .matlab_matrix import (
    MatlabMatrix,
    BlogcatalogDataset,
    WikipediaDataset,
    PPIDataset,
)
from .modelnet import (
    ModelNet10,
    ModelNet40,
    ModelNet10Train,
    ModelNet10Test,
    ModelNet40Train,
    ModelNet40Test,
)
from .utils import (
    get_label_number,
    random_splits_mask,
    random_splits_mask_class,
    graph_cross_validation,
    graph_set_fold_id,
    graph_random_splits,
    graph_get_split,
)


def build_dataset(args, path="~/.cache-autogl/"):
    path = osp.join(path, "data", args.dataset)
    path = os.path.expanduser(path)
    return DATASET_DICT[args.dataset](path)


def build_dataset_from_name(dataset_name, path="~/.cache-autogl/"):
    path = osp.join(path, "data", dataset_name)
    path = os.path.expanduser(path)
    dataset = DATASET_DICT[dataset_name](path)
    if "ogbn" in dataset_name:
        # dataset.data, dataset.slices = dataset.collate([dataset.data])
        # dataset.data.num_nodes = dataset.data.num_nodes[0]
        if dataset.data.y.shape[-1] == 1:
            dataset.data.y = torch.squeeze(dataset.data.y)
    return dataset


__all__ = [
    "register_dataset",
    "build_dataset",
    "build_dataset_from_name",
    "get_label_number",
    "random_splits_mask",
    "random_splits_mask_class",
    "graph_cross_validation",
    "graph_set_fold_id",
    "graph_random_splits",
    "graph_get_split",
    "AmazonComputersDataset",
    "AmazonPhotoDataset",
    "CoauthorPhysicsDataset",
    "CoauthorCSDataset",
    "CoraDataset",
    "CiteSeerDataset",
    "PubMedDataset",
    "RedditDataset",
    "MUTAGDataset",
    "IMDBBinaryDataset",
    "IMDBMultiDataset",
    "CollabDataset",
    "ProteinsDataset",
    "REDDITBinary",
    "REDDITMulti5K",
    "REDDITMulti12K",
    "PTCMRDataset",
    "NCI1Dataset",
    "ENZYMES",
    "QM9Dataset",
    "OGBNproductsDataset",
    "OGBNproteinsDataset",
    "OGBNarxivDataset",
    "OGBNpapers100MDataset",
    "OGBNmagDataset",
    "OGBGmolhivDataset",
    "OGBGmolpcbaDataset",
    "OGBGppaDataset",
    "OGBGcodeDataset",
    "OGBLppaDataset",
    "OGBLcollabDataset",
    "OGBLddiDataset",
    "OGBLcitationDataset",
    "OGBLwikikgDataset",
    "OGBLbiokgDataset",
    "GatneDataset",
    "AmazonDataset",
    "TwitterDataset",
    "YouTubeDataset",
    "GTNDataset",
    "ACM_GTNDataset",
    "DBLP_GTNDataset",
    "IMDB_GTNDataset",
    "HANDataset",
    "ACM_HANDataset",
    "DBLP_HANDataset",
    "IMDB_HANDataset",
    "MatlabMatrix",
    "BlogcatalogDataset",
    "WikipediaDataset",
    "PPIDataset",
    "ModelNet10",
    "ModelNet40",
    "ModelNet10Train",
    "ModelNet10Test",
    "ModelNet40Train",
    "ModelNet40Test",
]
