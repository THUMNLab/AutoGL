from autogl.module.train import NodeClassificationFullTrainer
from autogl.datasets import build_dataset_from_name
from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset

def test_node_trainer():

    dataset = build_dataset_from_name("cora")
    dataset = to_pyg_dataset(dataset)
    
    node_trainer = NodeClassificationFullTrainer(
        model='gcn',
        init=False,
        lr=1e-2,
        weight_decay=5e-4,
        max_epoch=200,
        early_stopping_round=200,
    )

    node_trainer.num_features = dataset[0].x.size(1)
    node_trainer.num_classes = dataset[0].y.max().item() + 1
    node_trainer.initialize()

    print(node_trainer.encoder.encoder)
    print(node_trainer.decoder.decoder)

    node_trainer.train(dataset, True)
    result = node_trainer.evaluate(dataset, "test", "acc")
    print("Acc:", result)

test_node_trainer()
