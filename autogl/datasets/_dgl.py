import os
import torch
import dgl

# from autogl.data.graph import GeneralStaticGraphGenerator
from autogl.data.graph.utils import conversion as _conversion
from autogl.data import InMemoryStaticGraphSet
from ._dataset_registry import DatasetUniversalRegistry


@DatasetUniversalRegistry.register_dataset("cora")
class CoraDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.CoraGraphDataset(
            os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(CoraDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(CoraDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label'],
        #                 'train_mask': dgl_graph.ndata['train_mask'],
        #                 'val_mask': dgl_graph.ndata['val_mask'],
        #                 'test_mask': dgl_graph.ndata['test_mask']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("CiteSeer".lower())
class CiteSeerDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.CiteseerGraphDataset(
            os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(CiteSeerDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(CiteSeerDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label'],
        #                 'train_mask': dgl_graph.ndata['train_mask'],
        #                 'val_mask': dgl_graph.ndata['val_mask'],
        #                 'test_mask': dgl_graph.ndata['test_mask']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("PubMed".lower())
class PubMedDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.PubmedGraphDataset(
            os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(PubMedDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(PubMedDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label'],
        #                 'train_mask': dgl_graph.ndata['train_mask'],
        #                 'val_mask': dgl_graph.ndata['val_mask'],
        #                 'test_mask': dgl_graph.ndata['test_mask']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("reddit")
class RedditDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.RedditDataset(
            raw_dir=os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(RedditDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(RedditDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label'],
        #                 'train_mask': dgl_graph.ndata['train_mask'],
        #                 'val_mask': dgl_graph.ndata['val_mask'],
        #                 'test_mask': dgl_graph.ndata['test_mask']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("amazon_computers")
class AmazonComputersDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.AmazonCoBuyComputerDataset(
            raw_dir=os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(AmazonComputersDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(AmazonComputersDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("amazon_photo")
class AmazonPhotoDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.AmazonCoBuyPhotoDataset(
            raw_dir=os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(AmazonPhotoDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(AmazonPhotoDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("coauthor_physics")
class CoauthorPhysicsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.CoauthorPhysicsDataset(
            raw_dir=os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(CoauthorPhysicsDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(CoauthorPhysicsDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("coauthor_cs")
class CoauthorCSDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.CoauthorCSDataset(
            raw_dir=os.path.join(path, '_dgl')
        )
        dgl_graph: dgl.DGLGraph = dgl_dataset[0]
        super(CoauthorCSDataset, self).__init__(
            [_conversion.dgl_graph_to_general_static_graph(dgl_graph)]
        )
        # super(CoauthorCSDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'feat': dgl_graph.ndata['feat'],
        #                 'label': dgl_graph.ndata['label']
        #             },
        #             torch.vstack(dgl_graph.edges())
        #         )
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("mutag")
class MUTAGDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "MUTAG", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(MUTAGDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(MUTAGDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("enzymes")
class ENZYMESDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.TUDataset(
            "ENZYMES", raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['node_attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['node_labels']
            del dgl_graph.ndata['node_attr']
            del dgl_graph.ndata['node_labels']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(ENZYMESDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(ENZYMESDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'node_labels': dgl_graph.ndata['node_labels'],
        #                 'node_attr': dgl_graph.ndata['node_attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': label}
        #         ) for (dgl_graph, label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("imdb-b")
@DatasetUniversalRegistry.register_dataset("imdb-binary")
@DatasetUniversalRegistry.register_dataset("IMDBb".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary".upper())
@DatasetUniversalRegistry.register_dataset("IMDBBinary")
class IMDBBinaryDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "IMDBBINARY", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(IMDBBinaryDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(IMDBBinaryDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("imdb-m")
@DatasetUniversalRegistry.register_dataset("imdb-multi")
@DatasetUniversalRegistry.register_dataset("IMDBm".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti".upper())
@DatasetUniversalRegistry.register_dataset("IMDBMulti")
class IMDBMultiDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "IMDBMULTI", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(IMDBMultiDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(IMDBMultiDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("reddit-b")
@DatasetUniversalRegistry.register_dataset("reddit-binary")
@DatasetUniversalRegistry.register_dataset("RedditB".upper())
@DatasetUniversalRegistry.register_dataset("RedditBinary".upper())
class RedditBinaryDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "REDDITBINARY", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(RedditBinaryDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(RedditBinaryDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("reddit-multi-5k")
@DatasetUniversalRegistry.register_dataset("RedditMulti5K".upper())
class REDDITMulti5KDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "REDDITMULTI5K", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(REDDITMulti5KDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )

        # super(REDDITMulti5KDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("collab")
class COLLABDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "COLLAB", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(COLLABDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(COLLABDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("proteins")
class ProteinsDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "PROTEINS", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(ProteinsDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(ProteinsDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("ptc-mr")
class PTCMRDataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "PTC", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(PTCMRDataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(PTCMRDataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )


@DatasetUniversalRegistry.register_dataset("nci1")
class NCI1Dataset(InMemoryStaticGraphSet):
    def __init__(self, path: str):
        dgl_dataset = dgl.data.GINDataset(
            "NCI1", False, raw_dir=os.path.join(path, "_dgl")
        )

        def _transform(dgl_graph: dgl.DGLGraph, label: torch.Tensor):
            dgl_graph.ndata['feat'] = dgl_graph.ndata['attr']
            dgl_graph.ndata['node_label'] = dgl_graph.ndata['label']
            del dgl_graph.ndata['attr']
            del dgl_graph.ndata['label']
            static_graph = _conversion.dgl_graph_to_general_static_graph(dgl_graph)
            static_graph.data['label'] = label
            return static_graph

        super(NCI1Dataset, self).__init__(
            [_transform(dgl_graph, label) for (dgl_graph, label) in dgl_dataset]
        )
        # super(NCI1Dataset, self).__init__(
        #     [
        #         GeneralStaticGraphGenerator.create_homogeneous_static_graph(
        #             {
        #                 'label': dgl_graph.ndata['label'],
        #                 'attr': dgl_graph.ndata['attr']
        #             },
        #             torch.vstack(dgl_graph.edges()),
        #             graph_data={'label': graph_label}
        #         )
        #         for (dgl_graph, graph_label) in dgl_dataset
        #     ]
        # )
