import os
import torch
import dgl

# from autogl.data.graph import GeneralStaticGraphGenerator
from autogl.data.graph.utils import conversion as _conversion
from autogl.data import InMemoryStaticGraphSet
from ._dataset_registry import DatasetUniversalRegistry

@DatasetUniversalRegistry.register_dataset("cora")
def get_cora_dataset(path, *args, **kwargs):
    return dgl.data.CoraGraphDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("citeseer")
def get_citeseer_dataset(path, *args, **kwargs):
    return dgl.data.CiteseerGraphDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("pubmed")
def get_pubmed_dataset(path, *args, **kwargs):
    return dgl.data.PubmedGraphDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit")
def get_reddit_dataset(path, *args, **kwargs):
    return dgl.data.RedditDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("amazon_computers")
def get_amazon_computers_dataset(path, *args, **kwargs):
    return dgl.data.AmazonCoBuyComputerDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("amazon_photo")
def get_amazon_photo_dataset(path, *args, **kwargs):
    return dgl.data.AmazonCoBuyPhotoDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("coauthor_physics")
def get_coauthor_physics_dataset(path, *args, **kwargs):
    return dgl.data.CoauthorPhysicsDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("coauthor_cs")
def get_coauthor_cs_dataset(path, *args, **kwargs):
    return dgl.data.CoauthorCSDataset(os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("mutag")
def get_mutag_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("MUTAG", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("enzymes")
def get_enzymes_dataset(path, *args, **kwargs):
    return dgl.data.TUDataset("ENZYMES", raw_dir=os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("imdb-b")
@DatasetUniversalRegistry.register_dataset("imdb-binary")
@DatasetUniversalRegistry.register_dataset("IMDBb".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary")
def get_imdb_binary_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("IMDBBINARY", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("imdb-m")
@DatasetUniversalRegistry.register_dataset("imdb-multi")
@DatasetUniversalRegistry.register_dataset("IMDBm".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti")
def get_imdb_multi_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("IMDBMULTI", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-b")
@DatasetUniversalRegistry.register_dataset("reddit-binary")
@DatasetUniversalRegistry.register_dataset("RedditB".upper())
@DatasetUniversalRegistry.register_dataset("RedditBinary".upper())
def get_reddit_binary_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("REDDITBINARY", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-multi-5k")
@DatasetUniversalRegistry.register_dataset("RedditMulti5K".upper())
def get_reddit_multi5k_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("REDDITMULTI5K", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("reddit-multi-12k")
@DatasetUniversalRegistry.register_dataset("RedditMulti12K".upper())
def get_reddit_multi12k_dataset(path, *args, **kwargs):
    return dgl.data.TUDataset("REDDIT-MULTI-12K", raw_dir=os.path.join(path, '_dgl'), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("collab")
def get_collab_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("COLLAB", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("proteins")
def get_proteins_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("PROTEINS", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("ptc-mr")
def get_ptcmr_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("PTC", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)

@DatasetUniversalRegistry.register_dataset("nci1")
def get_nci1_dataset(path, *args, **kwargs):
    self_loop = kwargs.pop('self_loop', False)
    return dgl.data.GINDataset("NCI1", self_loop, raw_dir=os.path.join(path, "_dgl"), *args, **kwargs)
