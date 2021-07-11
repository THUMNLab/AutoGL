import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from . import register_dataset
from .utils import index_to_mask
from torch_geometric.data import Data


# OGBN


@register_dataset("ogbn-products")
class OGBNproductsDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-products"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path)
        super(OGBNproductsDataset, self).__init__(dataset, path)
        # Pre-compute GCN normalization.
        # adj_t = self.data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # self.data.adj_t = adj_t

        setattr(OGBNproductsDataset, "metric", "Accuracy")
        setattr(OGBNproductsDataset, "loss", "nll_loss")
        split_idx = self.get_idx_split()
        datalist = []
        for d in self:
            setattr(d, "train_mask", index_to_mask(split_idx["train"], d.y.shape[0]))
            setattr(d, "val_mask", index_to_mask(split_idx["valid"], d.y.shape[0]))
            setattr(d, "test_mask", index_to_mask(split_idx["test"], d.y.shape[0]))
            datalist.append(d)
        self.data, self.slices = self.collate(datalist)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBNproductsDataset, self).get(idx)


@register_dataset("ogbn-proteins")
class OGBNproteinsDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-proteins"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path)
        super(OGBNproteinsDataset, self).__init__(dataset, path)
        dataset_t = PygNodePropPredDataset(
            name=dataset, root=path, transform=T.ToSparseTensor()
        )

        # Move edge features to node features.
        self.data.x = dataset_t[0].adj_t.mean(dim=1)
        # dataset_t[0].adj_t.set_value_(None)
        del dataset_t

        setattr(OGBNproteinsDataset, "metric", "ROC-AUC")
        setattr(OGBNproteinsDataset, "loss", "binary_cross_entropy_with_logits")
        split_idx = self.get_idx_split()
        datalist = []
        for d in self:
            setattr(d, "train_mask", index_to_mask(split_idx["train"], d.y.shape[0]))
            setattr(d, "val_mask", index_to_mask(split_idx["valid"], d.y.shape[0]))
            setattr(d, "test_mask", index_to_mask(split_idx["test"], d.y.shape[0]))
            datalist.append(d)
        self.data, self.slices = self.collate(datalist)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBNproteinsDataset, self).get(idx)


@register_dataset("ogbn-arxiv")
class OGBNarxivDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-arxiv"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path)
        super(OGBNarxivDataset, self).__init__(dataset, path)
        setattr(OGBNarxivDataset, "metric", "Accuracy")
        setattr(OGBNarxivDataset, "loss", "nll_loss")
        split_idx = self.get_idx_split()

        datalist = []
        for d in self:
            setattr(d, "train_mask", index_to_mask(split_idx["train"], d.y.shape[0]))
            setattr(d, "val_mask", index_to_mask(split_idx["valid"], d.y.shape[0]))
            setattr(d, "test_mask", index_to_mask(split_idx["test"], d.y.shape[0]))
            datalist.append(d)
        self.data, self.slices = self.collate(datalist)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBNarxivDataset, self).get(idx)


@register_dataset("ogbn-papers100M")
class OGBNpapers100MDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-papers100M"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path)
        super(OGBNpapers100MDataset, self).__init__(dataset, path)
        setattr(OGBNpapers100MDataset, "metric", "Accuracy")
        setattr(OGBNpapers100MDataset, "loss", "nll_loss")
        split_idx = self.get_idx_split()
        datalist = []
        for d in self:
            setattr(d, "train_mask", index_to_mask(split_idx["train"], d.y.shape[0]))
            setattr(d, "val_mask", index_to_mask(split_idx["valid"], d.y.shape[0]))
            setattr(d, "test_mask", index_to_mask(split_idx["test"], d.y.shape[0]))
            datalist.append(d)
        self.data, self.slices = self.collate(datalist)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBNpapers100MDataset, self).get(idx)


@register_dataset("ogbn-mag")
class OGBNmagDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-mag"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path)
        super(OGBNmagDataset, self).__init__(dataset, path)

        # Preprocess
        rel_data = self[0]
        # We are only interested in paper <-> paper relations.
        self.data = Data(
            x=rel_data.x_dict["paper"],
            edge_index=rel_data.edge_index_dict[("paper", "cites", "paper")],
            y=rel_data.y_dict["paper"],
        )

        # self.data = T.ToSparseTensor()(data)
        # self[0].adj_t = self[0].adj_t.to_symmetric()

        setattr(OGBNmagDataset, "metric", "Accuracy")
        setattr(OGBNmagDataset, "loss", "nll_loss")
        split_idx = self.get_idx_split()

        datalist = []
        for d in self:
            setattr(d, "train_mask", index_to_mask(split_idx["train"], d.y.shape[0]))
            setattr(d, "val_mask", index_to_mask(split_idx["valid"], d.y.shape[0]))
            setattr(d, "test_mask", index_to_mask(split_idx["test"], d.y.shape[0]))
            datalist.append(d)
        self.data, self.slices = self.collate(datalist)

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBNmagDataset, self).get(idx)


# OGBG


@register_dataset("ogbg-molhiv")
class OGBGmolhivDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-molhiv"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGmolhivDataset, self).__init__(dataset, path)
        setattr(OGBGmolhivDataset, "metric", "ROC-AUC")
        setattr(OGBGmolhivDataset, "loss", "binary_cross_entropy_with_logits")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBGmolhivDataset, self).get(idx)


@register_dataset("ogbg-molpcba")
class OGBGmolpcbaDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-molpcba"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGmolpcbaDataset, self).__init__(dataset, path)
        setattr(OGBGmolpcbaDataset, "metric", "AP")
        setattr(OGBGmolpcbaDataset, "loss", "binary_cross_entropy_with_logits")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBGmolpcbaDataset, self).get(idx)


@register_dataset("ogbg-ppa")
class OGBGppaDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-ppa"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGppaDataset, self).__init__(dataset, path)
        setattr(OGBGppaDataset, "metric", "Accuracy")
        setattr(OGBGppaDataset, "loss", "cross_entropy")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBGppaDataset, self).get(idx)


@register_dataset("ogbg-code")
class OGBGcodeDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-code"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGcodeDataset, self).__init__(dataset, path)
        setattr(OGBGcodeDataset, "metric", "F1 score")
        setattr(OGBGcodeDataset, "loss", "cross_entropy")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBGcodeDataset, self).get(idx)


# OGBL


@register_dataset("ogbl-ppa")
class OGBLppaDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-ppa"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLppaDataset, self).__init__(dataset, path)
        setattr(OGBLppaDataset, "metric", "Hits@100")
        setattr(OGBLppaDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLppaDataset, self).get(idx)


@register_dataset("ogbl-collab")
class OGBLcollabDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-collab"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLcollabDataset, self).__init__(dataset, path)
        setattr(OGBLcollabDataset, "metric", "Hits@50")
        setattr(OGBLcollabDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLcollabDataset, self).get(idx)


@register_dataset("ogbl-ddi")
class OGBLddiDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-ddi"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLddiDataset, self).__init__(dataset, path)
        setattr(OGBLddiDataset, "metric", "Hits@20")
        setattr(OGBLddiDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLddiDataset, self).get(idx)


@register_dataset("ogbl-citation")
class OGBLcitationDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-citation"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLcitationDataset, self).__init__(dataset, path)
        setattr(OGBLcitationDataset, "metric", "MRR")
        setattr(OGBLcitationDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLcitationDataset, self).get(idx)


@register_dataset("ogbl-wikikg")
class OGBLwikikgDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-wikikg"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLwikikgDataset, self).__init__(dataset, path)
        setattr(OGBLwikikgDataset, "metric", "MRR")
        setattr(OGBLwikikgDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLwikikgDataset, self).get(idx)


@register_dataset("ogbl-biokg")
class OGBLbiokgDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-biokg"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path)
        super(OGBLbiokgDataset, self).__init__(dataset, path)
        setattr(OGBLbiokgDataset, "metric", "MRR")
        setattr(OGBLbiokgDataset, "loss", "pos_neg_loss")

    def get(self, idx):
        if hasattr(self, "__data_list__"):
            delattr(self, "__data_list__")
        if hasattr(self, "_data_list"):
            delattr(self, "_data_list")
        return super(OGBLbiokgDataset, self).get(idx)
