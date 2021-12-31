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
class CoraDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Planetoid(os.path.join(path, '_pyg'), "Cora")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]

        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {
                'x': pyg_data.x,
                'y': pyg_data.y,
                'train_mask': getattr(pyg_data, 'train_mask'),
                'val_mask': getattr(pyg_data, 'val_mask'),
                'test_mask': getattr(pyg_data, 'test_mask')
            },
            pyg_data.edge_index
        )
        super(CoraDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("CiteSeer".lower())
class CiteSeerDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Planetoid(os.path.join(path, '_pyg'), "CiteSeer")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]

        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {
                'x': pyg_data.x,
                'y': pyg_data.y,
                'train_mask': getattr(pyg_data, 'train_mask'),
                'val_mask': getattr(pyg_data, 'val_mask'),
                'test_mask': getattr(pyg_data, 'test_mask')
            },
            pyg_data.edge_index
        )
        super(CiteSeerDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("PubMed".lower())
class PubMedDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Planetoid(os.path.join(path, '_pyg'), "PubMed")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]

        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {
                'x': pyg_data.x,
                'y': pyg_data.y,
                'train_mask': getattr(pyg_data, 'train_mask'),
                'val_mask': getattr(pyg_data, 'val_mask'),
                'test_mask': getattr(pyg_data, 'test_mask')
            },
            pyg_data.edge_index
        )
        super(PubMedDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("flickr")
class FlickrDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Flickr(os.path.join(path, '_pyg'))
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]

        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {
                'x': pyg_data.x,
                'y': pyg_data.y,
                'train_mask': getattr(pyg_data, 'train_mask'),
                'val_mask': getattr(pyg_data, 'val_mask'),
                'test_mask': getattr(pyg_data, 'test_mask')
            },
            pyg_data.edge_index
        )
        super(FlickrDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("reddit")
class RedditDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Reddit(os.path.join(path, '_pyg'))
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]

        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {
                'x': pyg_data.x,
                'y': pyg_data.y,
                'train_mask': getattr(pyg_data, 'train_mask'),
                'val_mask': getattr(pyg_data, 'val_mask'),
                'test_mask': getattr(pyg_data, 'test_mask')
            },
            pyg_data.edge_index
        )
        super(RedditDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("amazon_computers")
class AmazonComputersDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Amazon(os.path.join(path, '_pyg'), "Computers")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]
        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {'x': pyg_data.x, 'y': pyg_data.y},
            pyg_data.edge_index
        )
        super(AmazonComputersDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("amazon_photo")
class AmazonPhotoDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Amazon(os.path.join(path, '_pyg'), "Photo")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]
        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {'x': pyg_data.x, 'y': pyg_data.y},
            pyg_data.edge_index
        )
        super(AmazonPhotoDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("coauthor_physics")
class CoauthorPhysicsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Coauthor(os.path.join(path, '_pyg'), "Physics")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]
        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {'x': pyg_data.x, 'y': pyg_data.y},
            pyg_data.edge_index
        )
        super(CoauthorPhysicsDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("coauthor_cs")
class CoauthorCSDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = Coauthor(os.path.join(path, '_pyg'), "CS")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        pyg_data = pyg_dataset[0]
        static_graph = GeneralStaticGraphGenerator.create_homogeneous_static_graph(
            {'x': pyg_data.x, 'y': pyg_data.y},
            pyg_data.edge_index
        )
        super(CoauthorCSDataset, self).__init__([static_graph])


@DatasetUniversalRegistry.register_dataset("ppi")
class PPIDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        train_dataset = PPI(os.path.join(path, '_pyg'), 'train')
        if hasattr(train_dataset, "__data_list__"):
            delattr(train_dataset, "__data_list__")
        if hasattr(train_dataset, "_data_list"):
            delattr(train_dataset, "_data_list")
        val_dataset = PPI(os.path.join(path, '_pyg'), 'val')
        if hasattr(val_dataset, "__data_list__"):
            delattr(val_dataset, "__data_list__")
        if hasattr(val_dataset, "_data_list"):
            delattr(val_dataset, "_data_list")
        test_dataset = PPI(os.path.join(path, '_pyg'), 'test')
        if hasattr(test_dataset, "__data_list__"):
            delattr(test_dataset, "__data_list__")
        if hasattr(test_dataset, "_data_list"):
            delattr(test_dataset, "_data_list")
        train_index = range(len(train_dataset))
        val_index = range(len(train_dataset), len(train_dataset) + len(val_dataset))
        test_index = range(
            len(train_dataset) + len(val_dataset),
            len(train_dataset) + len(val_dataset) + len(test_dataset)
        )
        super(PPIDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': data.x, 'y': data.y}, data.edge_index
                ) for data in train_dataset
            ] +
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': data.x, 'y': data.y}, data.edge_index
                ) for data in val_dataset
            ] +
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': data.x, 'y': data.y}, data.edge_index
                ) for data in test_dataset
            ],
            train_index, val_index, test_index
        )


@DatasetUniversalRegistry.register_dataset("qm9")
class QM9Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = QM9(os.path.join(path, '_pyg'))
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(QM9Dataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': data.x, 'pos': data.pos, 'z': data.z},
                    data.edge_index,
                    edges_data={'edge_attr': data.edge_attr},
                    graph_data={'idx': data.idx, 'y': data.y}
                ) for data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("mutag")
class MUTAGDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "MUTAG")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(MUTAGDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index,
                    edges_data={'edge_attr': pyg_data.edge_attr},
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("enzymes")
class ENZYMESDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "ENZYMES")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ENZYMESDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("imdb-b")
@DatasetUniversalRegistry.register_dataset("imdb-binary")
@DatasetUniversalRegistry.register_dataset("IMDBb".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary")
class IMDBBinaryDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "IMDB-BINARY")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(IMDBBinaryDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("imdb-m")
@DatasetUniversalRegistry.register_dataset("imdb-multi")
@DatasetUniversalRegistry.register_dataset("IMDBm".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti")
class IMDBMultiDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "IMDB-MULTI")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(IMDBMultiDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("reddit-b")
@DatasetUniversalRegistry.register_dataset("reddit-binary")
@DatasetUniversalRegistry.register_dataset("RedditB".upper())
@DatasetUniversalRegistry.register_dataset("RedditBinary".upper())
class RedditBinaryDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "REDDIT-BINARY")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(RedditBinaryDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("reddit-multi-5k")
@DatasetUniversalRegistry.register_dataset("RedditMulti5K".upper())
class REDDITMulti5KDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "REDDIT-MULTI-5K")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(REDDITMulti5KDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("reddit-multi-12k")
@DatasetUniversalRegistry.register_dataset("RedditMulti12K".upper())
class REDDITMulti12KDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "REDDIT-MULTI-12K")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(REDDITMulti12KDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("collab")
class COLLABDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "COLLAB")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(COLLABDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("proteins")
class ProteinsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "PROTEINS")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ProteinsDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index, graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("ptc-mr")
class PTCMRDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "PTC_MR")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(PTCMRDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index,
                    edges_data={'edge_attr': pyg_data.edge_attr},
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("nci1")
class NCI1Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "NCI1")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(NCI1Dataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("nci109")
class NCI109Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = TUDataset(os.path.join(path, '_pyg'), "NCI109")
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(NCI109Dataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'x': pyg_data.x}, pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("ModelNet10Training")
class ModelNet10TrainingDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = ModelNet(
            os.path.join(path, '_pyg'), '10', True,
            pre_transform=torch_geometric.transforms.FaceToEdge()
        )
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ModelNet10TrainingDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'pos': pyg_data.pos},
                    pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("ModelNet10Test")
class ModelNet10TestDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = ModelNet(
            os.path.join(path, '_pyg'), '10', False,
            pre_transform=torch_geometric.transforms.FaceToEdge()
        )
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ModelNet10TestDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'pos': pyg_data.pos},
                    pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("ModelNet40Training")
class ModelNet40TrainingDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = ModelNet(
            os.path.join(path, '_pyg'), '40', True,
            pre_transform=torch_geometric.transforms.FaceToEdge()
        )
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ModelNet40TrainingDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'pos': pyg_data.pos},
                    pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )


@DatasetUniversalRegistry.register_dataset("ModelNet40Test")
class ModelNet40TestDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        pyg_dataset = ModelNet(
            os.path.join(path, '_pyg'), '40', False,
            pre_transform=torch_geometric.transforms.FaceToEdge()
        )
        if hasattr(pyg_dataset, "__data_list__"):
            delattr(pyg_dataset, "__data_list__")
        if hasattr(pyg_dataset, "_data_list"):
            delattr(pyg_dataset, "_data_list")
        super(ModelNet40TestDataset, self).__init__(
            [
                GeneralStaticGraphGenerator.create_homogeneous_static_graph(
                    {'pos': pyg_data.pos},
                    pyg_data.edge_index,
                    graph_data={'y': pyg_data.y}
                )
                for pyg_data in pyg_dataset
            ]
        )
