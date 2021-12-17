from autogl.module.train import GraphClassificationFullTrainer
from autogl.datasets import build_dataset_from_name, utils
from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset

def test_graph_trainer():

    dataset = build_dataset_from_name("mutag")
    utils.graph_random_splits(dataset, 0.8, 0.1)
    dataset = to_pyg_dataset(dataset)
    
    lp_trainer = GraphClassificationFullTrainer(model='gin', init=False)

    lp_trainer.num_features = dataset[0].x.size(1)
    lp_trainer.num_classes = max([d.y for d in dataset]).item() + 1
    lp_trainer.num_graph_features = 0
    lp_trainer.initialize()

    print(lp_trainer.encoder.encoder)
    print(lp_trainer.decoder.decoder)

    lp_trainer.train(dataset, True)
    result = lp_trainer.evaluate(dataset, "test", "acc")
    print("Acc:", result)

test_graph_trainer()
