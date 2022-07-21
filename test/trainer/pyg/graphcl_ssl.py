from autogl.module.train.ssl import GraphCLSemisupervisedTrainer
from autogl.datasets import build_dataset_from_name, utils
from autogl.datasets.utils.conversion._to_pyg_dataset import to_pyg_dataset
from autogl.module.model.encoders.base_encoder import AutoHomogeneousEncoderMaintainer
from autogl.module.model.decoders import BaseDecoderMaintainer

def test_graph_trainer():
    dataset = build_dataset_from_name("nci1")
    utils.graph_random_splits(dataset, 0.1, 0)
    dataset = to_pyg_dataset(dataset)

    num_features = dataset[0].x.size(1)
    num_classes = max([d.y for d in dataset]).item() + 1
    num_graph_features = 0

    trainer = GraphCLSemisupervisedTrainer(
        model=('gcn', 'sumpoolmlp'),
        prediction_model_head='sumpoolmlp',
        views_fn=["random2", "random2"],
        p_early_stopping_round=50,
        f_early_stopping_round=50,
        init=False
    )

    trainer.num_features = num_features
    trainer.num_classes = num_classes
    trainer.num_graph_features = num_graph_features
    print(f"{num_features}#{num_classes}#{num_graph_features}")
    trainer.initialize()
    print("Stage 1 ...")

    assert trainer.num_features == num_features
    assert trainer.num_classes == num_classes
    assert trainer.num_graph_features == num_graph_features
    assert trainer.encoder.input_dimension == num_features
    assert trainer.prediction_model_head.output_dimension == num_classes
    print("Stage 1 over ...")

    print(trainer.encoder.encoder)
    print(trainer.decoder.decoder)
    print(trainer.prediction_model_head.decoder)

    print("Stage 2 ...")
    trainer.train(dataset, True)
    result = trainer.evaluate(dataset, "test", "acc")
    print("Stage 2 over ...")
    print("Acc:", result)

if __name__ == "__main__":
    test_graph_trainer()
