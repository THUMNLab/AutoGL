import os
from autogl.data.graph import GeneralStaticGraphGenerator
from autogl.data import InMemoryStaticGraphSet
from ._dataset_registry import DatasetUniversalRegistry
import torch_geometric
from torch_geometric.datasets import (
    Amazon, Coauthor, Flickr, ModelNet,
    Planetoid, PPI, QM9, Reddit, TUDataset
)

@DatasetUniversalRegistry.register_dataset("cora")
def get_cora_dataset(path, *args, **kwargs):
    return Planetoid(os.path.join(path, '_pyg'), "Cora", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("CiteSeer".lower())
def get_citeseer_dataset(path, *args, **kwargs):
    return Planetoid(os.path.join(path, '_pyg'), "Citeseer", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("PubMed".lower())
def get_pubmed_dataset(path, *args, **kwargs):
    return Planetoid(os.path.join(path, '_pyg'), "PubMed", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("flickr".lower())
def get_flickr_dataset(path, *args, **kwargs):
    return Flickr(os.path.join(path, '_pyg'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit".lower())
def get_reddit_dataset(path, *args, **kwargs):
    return Reddit(os.path.join(path, '_pyg'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("amazon_computers".lower())
def get_amazon_computers_dataset(path, *args, **kwargs):
    return Amazon(os.path.join(path, '_pyg'), "Computers", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("amazon_photo".lower())
def get_amazon_photo_dataset(path, *args, **kwargs):
    return Amazon(os.path.join(path, '_pyg'), "Photo", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("coauthor_physics".lower())
def get_coauthor_physics_dataset(path, *args, **kwargs):
    return Coauthor(os.path.join(path, '_pyg'), 'Physics', *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("coauthor_cs".lower())
def get_coauthor_cs_dataset(path, *args, **kwargs):
    return Coauthor(os.path.join(path, '_pyg'), "CS", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ppi".lower())
def get_ppi_dataset(path, *args, **kwargs):
    return PPI(os.path.join(path, '_pyg'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("qm9".lower())
def get_qm9_dataset(path, *args, **kwargs):
    return QM9(os.path.join(path, '_pyg'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("mutag".lower())
def get_mutag_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "MUTAG", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("enzymes".lower())
def get_enzymes_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "ENZYMES", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("imdb-b")
@DatasetUniversalRegistry.register_dataset("imdb-binary")
@DatasetUniversalRegistry.register_dataset("IMDBb".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary")
def get_imdb_binary_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "IMDB-BINARY", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("imdb-m")
@DatasetUniversalRegistry.register_dataset("imdb-multi")
@DatasetUniversalRegistry.register_dataset("IMDBm".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti")
def get_imdb_multi_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "IMDB-MULTI", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-b")
@DatasetUniversalRegistry.register_dataset("reddit-binary")
@DatasetUniversalRegistry.register_dataset("RedditB".upper())
@DatasetUniversalRegistry.register_dataset("RedditBinary".upper())
def get_reddit_binary_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "REDDIT-BINARY", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-multi-5k")
@DatasetUniversalRegistry.register_dataset("RedditMulti5K".upper())
def get_reddit_multi5k_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "REDDIT-MULTI-5K", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-multi-12k")
@DatasetUniversalRegistry.register_dataset("RedditMulti12K".upper())
def get_reddit_multi12k_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "REDDIT-MULTI-12K", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("collab")
def get_collab_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "COLLAB", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("proteins")
def get_proteins_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "PROTEINS", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ptc-mr")
def get_ptcmr_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "PTC_MR", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("nci1")
def get_nci1_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "NCI1", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("nci109")
def get_nci109_dataset(path, *args, **kwargs):
    return TUDataset(os.path.join(path, '_pyg'), "NCI109", *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ModelNet".lower())
def get_modelnet_dataset(path, *args, **kwargs):
    return ModelNet(os.path.join(path, '_pyg'), *args, **kwargs)
