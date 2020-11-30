import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from . import register_dataset

# OGBN


@register_dataset("ogbn-products")
class OGBNproductsDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-products"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBNproductsDataset, self).__init__(
            dataset, path, transform=T.ToSparseTensor()
        )
        setattr(OGBNproductsDataset, "metric", "Accuracy")
        setattr(OGBNproductsDataset, "loss", "nll_loss")


@register_dataset("ogbn-proteins")
class OGBNproteinsDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-proteins"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBNproteinsDataset, self).__init__(
            dataset, path, transform=T.ToSparseTensor()
        )
        setattr(OGBNproteinsDataset, "metric", "ROC-AUC")
        setattr(OGBNproteinsDataset, "loss", "BCEWithLogitsLoss")


@register_dataset("ogbn-arxiv")
class OGBNarxivDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-arxiv"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBNarxivDataset, self).__init__(
            dataset, path, transform=T.ToSparseTensor()
        )
        setattr(OGBNarxivDataset, "metric", "Accuracy")
        setattr(OGBNarxivDataset, "loss", "nll_loss")


@register_dataset("ogbn-papers100M")
class OGBNpapers100MDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-papers100M"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBNpapers100MDataset, self).__init__(
            dataset, path, transform=T.ToSparseTensor()
        )
        setattr(OGBNpapers100MDataset, "metric", "Accuracy")
        setattr(OGBNpapers100MDataset, "loss", "nll_loss")


@register_dataset("ogbn-mag")
class OGBNmagDataset(PygNodePropPredDataset):
    def __init__(self, path):
        dataset = "ogbn-mag"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygNodePropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBNmagDataset, self).__init__(
            dataset, path, transform=T.ToSparseTensor()
        )
        setattr(OGBNmagDataset, "metric", "Accuracy")
        setattr(OGBNmagDataset, "loss", "nll_loss")


# OGBG


@register_dataset("ogbg-molhiv")
class OGBGmolhivDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-molhiv"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGmolhivDataset, self).__init__(dataset, path)
        setattr(OGBGmolhivDataset, "metric", "ROC-AUC")
        setattr(OGBGmolhivDataset, "loss", "BCEWithLogitsLoss")


@register_dataset("ogbg-molpcba")
class OGBGmolpcbaDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-molpcba"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGmolpcbaDataset, self).__init__(dataset, path)
        setattr(OGBGmolpcbaDataset, "metric", "AP")
        setattr(OGBGmolpcbaDataset, "loss", "BCEWithLogitsLoss")


@register_dataset("ogbg-ppa")
class OGBGppaDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-ppa"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGppaDataset, self).__init__(dataset, path)
        setattr(OGBGppaDataset, "metric", "Accuracy")
        setattr(OGBGppaDataset, "loss", "CrossEntropyLoss")


@register_dataset("ogbg-code")
class OGBGcodeDataset(PygGraphPropPredDataset):
    def __init__(self, path):
        dataset = "ogbg-code"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygGraphPropPredDataset(name=dataset, root=path)
        super(OGBGcodeDataset, self).__init__(dataset, path)
        setattr(OGBGcodeDataset, "metric", "F1 score")
        setattr(OGBGcodeDataset, "loss", "CrossEntropyLoss")


# OGBL


@register_dataset("ogbl-ppa")
class OGBLppaDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-ppa"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLppaDataset, self).__init__(dataset, path)
        setattr(OGBLppaDataset, "metric", "Hits@100")
        setattr(OGBLppaDataset, "loss", "pos_neg_loss")


@register_dataset("ogbl-collab")
class OGBLcollabDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-collab"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLcollabDataset, self).__init__(dataset, path)
        setattr(OGBLcollabDataset, "metric", "Hits@50")
        setattr(OGBLcollabDataset, "loss", "pos_neg_loss")


@register_dataset("ogbl-ddi")
class OGBLddiDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-ddi"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLddiDataset, self).__init__(dataset, path)
        setattr(OGBLddiDataset, "metric", "Hits@20")
        setattr(OGBLddiDataset, "loss", "pos_neg_loss")


@register_dataset("ogbl-citation")
class OGBLcitationDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-citation"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLcitationDataset, self).__init__(dataset, path)
        setattr(OGBLcitationDataset, "metric", "MRR")
        setattr(OGBLcitationDataset, "loss", "pos_neg_loss")


@register_dataset("ogbl-wikikg")
class OGBLwikikgDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-wikikg"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLwikikgDataset, self).__init__(dataset, path)
        setattr(OGBLwikikgDataset, "metric", "MRR")
        setattr(OGBLwikikgDataset, "loss", "pos_neg_loss")


@register_dataset("ogbl-biokg")
class OGBLbiokgDataset(PygLinkPropPredDataset):
    def __init__(self, path):
        dataset = "ogbl-biokg"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        PygLinkPropPredDataset(name=dataset, root=path, transform=T.ToSparseTensor())
        super(OGBLbiokgDataset, self).__init__(dataset, path)
        setattr(OGBLbiokgDataset, "metric", "MRR")
        setattr(OGBLbiokgDataset, "loss", "pos_neg_loss")
