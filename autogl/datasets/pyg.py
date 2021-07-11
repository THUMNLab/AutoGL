import os.path as osp

import torch

# import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Planetoid,
    Reddit,
    TUDataset,
    QM9,
    Amazon,
    Coauthor,
    Flickr,
)
from torch_geometric.utils import remove_self_loops
from . import register_dataset


@register_dataset("amazon_computers")
class AmazonComputersDataset(Amazon):
    def __init__(self, path):
        dataset = "Computers"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Amazon(path, dataset)
        super(AmazonComputersDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(AmazonComputersDataset, self).get(idx)


@register_dataset("amazon_photo")
class AmazonPhotoDataset(Amazon):
    def __init__(self, path):
        dataset = "Photo"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Amazon(path, dataset)
        super(AmazonPhotoDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(AmazonPhotoDataset, self).get(idx)


@register_dataset("coauthor_physics")
class CoauthorPhysicsDataset(Coauthor):
    def __init__(self, path):
        dataset = "Physics"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Coauthor(path, dataset)
        super(CoauthorPhysicsDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(CoauthorPhysicsDataset, self).get(idx)


@register_dataset("coauthor_cs")
class CoauthorCSDataset(Coauthor):
    def __init__(self, path):
        dataset = "CS"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Coauthor(path, dataset)
        super(CoauthorCSDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(CoauthorCSDataset, self).get(idx)


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self, path):
        dataset = "Cora"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Planetoid(path, dataset)
        super(CoraDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(CoraDataset, self).get(idx)


@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self, path):
        dataset = "CiteSeer"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Planetoid(path, dataset)
        super(CiteSeerDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(CiteSeerDataset, self).get(idx)


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self, path):
        dataset = "PubMed"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Planetoid(path, dataset)
        super(PubMedDataset, self).__init__(path, dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(PubMedDataset, self).get(idx)


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self, path):
        dataset = "Reddit"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        Reddit(path)
        super(RedditDataset, self).__init__(path)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(RedditDataset, self).get(idx)


@register_dataset("flickr")
class FlickrDataset(Flickr):
    def __init__(self, path):
        Flickr(path)
        super(FlickrDataset, self).__init__(path)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(FlickrDataset, self).get(idx)


@register_dataset("mutag")
class MUTAGDataset(TUDataset):
    def __init__(self, path):
        dataset = "MUTAG"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(MUTAGDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(MUTAGDataset, self).get(idx)


@register_dataset("imdb-b")
class IMDBBinaryDataset(TUDataset):
    def __init__(self, path):
        dataset = "IMDB-BINARY"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(IMDBBinaryDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(IMDBBinaryDataset, self).get(idx)


@register_dataset("imdb-m")
class IMDBMultiDataset(TUDataset):
    def __init__(self, path):
        dataset = "IMDB-MULTI"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(IMDBMultiDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(IMDBMultiDataset, self).get(idx)


@register_dataset("collab")
class CollabDataset(TUDataset):
    def __init__(self, path):
        dataset = "COLLAB"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(CollabDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(CollabDataset, self).get(idx)


@register_dataset("proteins")
class ProteinsDataset(TUDataset):
    def __init__(self, path):
        dataset = "PROTEINS"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(ProteinsDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ProteinsDataset, self).get(idx)


@register_dataset("reddit-b")
class REDDITBinary(TUDataset):
    def __init__(self, path):
        dataset = "REDDIT-BINARY"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(REDDITBinary, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(REDDITBinary, self).get(idx)


@register_dataset("reddit-multi-5k")
class REDDITMulti5K(TUDataset):
    def __init__(self, path):
        dataset = "REDDIT-MULTI-5K"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(REDDITMulti5K, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(REDDITMulti5K, self).get(idx)


@register_dataset("reddit-multi-12k")
class REDDITMulti12K(TUDataset):
    def __init__(self, path):
        dataset = "REDDIT-MULTI-12K"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(REDDITMulti12K, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(REDDITMulti12K, self).get(idx)


@register_dataset("ptc-mr")
class PTCMRDataset(TUDataset):
    def __init__(self, path):
        dataset = "PTC_MR"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(PTCMRDataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(PTCMRDataset, self).get(idx)


@register_dataset("nci1")
class NCI1Dataset(TUDataset):
    def __init__(self, path):
        dataset = "NCI1"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(NCI1Dataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(NCI1Dataset, self).get(idx)


@register_dataset("nci109")
class NCI109Dataset(TUDataset):
    def __init__(self, path):
        dataset = "NCI109"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(NCI109Dataset, self).__init__(path, name=dataset)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(NCI109Dataset, self).get(idx)


@register_dataset("enzymes")
class ENZYMES(TUDataset):
    def __init__(self, path):
        dataset = "ENZYMES"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        TUDataset(path, name=dataset)
        super(ENZYMES, self).__init__(path, name=dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data
            edge_nodes = data.edge_index.max() + 1
            if edge_nodes < data.x.size(0):
                data.x = data.x[:edge_nodes]
            return data
        else:
            return self.index_select(idx)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(ENZYMES, self).get(idx)


@register_dataset("qm9")
class QM9Dataset(QM9):
    def __init__(self, path):
        dataset = "QM9"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)

        target = 0

        class MyTransform(object):
            def __call__(self, data):
                # Specify target.
                data.y = data.y[:, target]
                return data

        class Complete(object):
            def __call__(self, data):
                device = data.edge_index.device
                row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
                col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
                row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
                col = col.repeat(data.num_nodes)
                edge_index = torch.stack([row, col], dim=0)
                edge_attr = None
                if data.edge_attr is not None:
                    idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                    size = list(data.edge_attr.size())
                    size[0] = data.num_nodes * data.num_nodes
                    edge_attr = data.edge_attr.new_zeros(size)
                    edge_attr[idx] = data.edge_attr
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                data.edge_attr = edge_attr
                data.edge_index = edge_index
                return data

        if not osp.exists(path):
            QM9(path)
        super(QM9Dataset, self).__init__(path)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(QM9Dataset, self).get(idx)
