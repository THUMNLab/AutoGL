import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from autogl.module.model import BaseAutoModel, BaseAutoEncoder, BaseEncoder, BaseDecoder, AutoClassifierDecoder
from autogl.module.model.encoders.base import AutoHomogeneousEncoder
from autogl.module.train import NodeClassificationFullTrainer
from autogl.utils.autobase import AutoModule

def activate(act, x):
    if hasattr(torch, act): return getattr(torch, act)(x)
    return getattr(F, act)(x)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.core = torch.nn.ModuleList([
            gnn.GCNConv(num_features, 16),
            gnn.GCNConv(16, num_classes)
        ])
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.core:
            x = layer(x, edge_index)
        return x

class AutoGCN(BaseAutoModel):
    def __init__(self, num_features=None, num_classes=None, device="cpu"):
        super().__init__(device=device)
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes
        self.hyper_parameter_space = []
        self.hyper_parameter = {}
    
    def initialize(self):
        self.model = GCN(self.num_features, self.num_classes).to(self.device)
    
    @property
    def num_features(self):
        return self.__num_features
    
    @num_features.setter
    def num_features(self, num_features):
        self.__num_features = num_features
    
    @property
    def num_classes(self):
        return self.__num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self.__num_classes = num_classes

    def from_hyper_parameter(self, hp):
        model = AutoGCN(self.num_features, self.num_classes, self.device)
        model.initialize()
        return model

class GCNEncoder(BaseEncoder):
    def __init__(self, num_features, last_dim, num_layers=2, hidden=(16,), dropout=0.6, act="relu"):
        super().__init__()
        self.core = torch.nn.ModuleList()
        
        # first layer
        if num_layers == 1:
            self.core.append(gnn.GCNConv(num_features, last_dim))
        else:
            self.core.append(gnn.GCNConv(num_features, hidden[0]))

        # middle layer
        for layer in range(num_layers - 2):
            self.core.append(gnn.GCNConv(hidden[layer], hidden[layer + 1]))

        # last layer
        if num_layers > 1:
            self.core.append(gnn.GCNConv(hidden[-1], last_dim))

        self.act = act
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = []
        for i, layer in enumerate(self.core):
            if i > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = activate(self.act, x)
            x = layer(x, edge_index)
            features.append(x)
        return features

class AutoGCNEncoder(AutoHomogeneousEncoder):
    def __init__(self, num_features=None, last_dim="auto", device="auto"):
        super().__init__(device)

        self.num_features = num_features

        self.hyper_parameter_space = [
            {
                "parameterName": "num_layers",
                "type": "DISCRETE",
                "feasiblePoints": "2,3,4",
            },
            {
                "parameterName": "hidden",
                "type": "NUMERICAL_LIST",
                "numericalType": "INTEGER",
                "length": 3,
                "minValue": [8, 8, 8],
                "maxValue": [128, 128, 128],
                "scalingType": "LOG",
                "cutPara": ("num_layers",),
                "cutFunc": lambda x: x[0] - 1,
            },
            {
                "parameterName": "dropout",
                "type": "DOUBLE",
                "maxValue": 0.8,
                "minValue": 0.2,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "act",
                "type": "CATEGORICAL",
                "feasiblePoints": ["leaky_relu", "relu", "elu", "tanh"],
            },
        ]

        self.hyper_parameter = {
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.6,
            "act": "tanh"
        }

        if last_dim == "auto":
            self.register_hyper_parameter_space({
                "parameterName": "last_dim",
                "type": "INTEGER",
                "scalingType": "LOG",
                "minValue": 8,
                "maxValue": 128
            })
            self.register_hyper_parameter("last_dim", 16)
        else:
            self.last_dim = last_dim
    
    def initialize(self):
        self.model = GCNEncoder(
            self.num_features, self.last_dim, self.num_layers, self.hidden, self.dropout, self.act
        )
        self.model.to(self.device)
    
    @property
    def num_features(self):
        return self.__num_features
    
    @num_features.setter
    def num_features(self, num_features):
        self.__num_features = num_features

    def from_hyper_parameter(self, hp):
        automodel = AutoGCNEncoder(self.num_features, self.last_dim, self.device)
        automodel.hyper_parameter = hp
        automodel.initialize()
        return automodel

class JKDecoder(BaseDecoder):
    def __init__(self, num_classes, input_dims):
        super().__init__()
        self.out = torch.nn.Linear(sum(input_dims), num_classes)

    def forward(self, features, data):
        return self.out(torch.cat(features, dim=1))

class AutoJKDecoder(AutoClassifierDecoder):
    def __init__(self, input_dim="auto", num_classes=None, device="auto"):
        super().__init__(device)
        self.num_classes = num_classes
    
    def initialize(self, encoder):
        self.model = JKDecoder(self.num_classes, [*encoder.hidden, encoder.last_dim])
        self.model.to(self.device)

    def from_hyper_parameter_and_encoder(self, hp, encoder):
        autodecoder = AutoJKDecoder(num_classes=self.num_classes)
        autodecoder.initialize(encoder)
        return autodecoder

    @property
    def num_classes(self):
        return self.__num_classes
    
    @num_classes.setter
    def num_classes(self, num_classes):
        self.__num_classes = num_classes

def test_auto_encoder():
    model = AutoGCNEncoder(last_dim="auto")
    model.num_features = 1222
    model.initialize()

    model2 = model.from_hyper_parameter({
        "num_layers": 3,
        "hidden": [16, 18],
        "dropout": 0.6,
        "act": "tanh",
        "last_dim": 120
    })

def test_auto_decoder():
    encoder = AutoGCNEncoder(last_dim="auto")
    decoder = AutoJKDecoder()
    encoder.num_features = 100
    decoder.num_classes = 7
    encoder.initialize()
    decoder.initialize(encoder)
    print(encoder.model)
    print(decoder.model)

def test_trainer_encoder_decoder():
    from autogl.datasets import build_dataset_from_name
    from autogl.datasets.utils.conversion import general_static_graphs_to_pyg_dataset
    cora = build_dataset_from_name("cora")
    cora = general_static_graphs_to_pyg_dataset(cora)
    data = cora[0]

    trainer = NodeClassificationFullTrainer(encoder=AutoGCNEncoder(), decoder=AutoJKDecoder(), init=False, device="auto", loss="cross_entropy")

    # support setting the property after instantiate
    trainer.num_features = data.x.size(1)
    trainer.num_classes = data.y.max() + 1

    # support duplicate from hyper parameter
    spaces = trainer.combined_hyper_parameter_space()
    print("space of trainer")
    print(spaces)

    trainer = trainer.duplicate_from_hyper_parameter({
        "trainer": {
            "max_epoch": 200,
            "early_stopping_round": 200,
            "lr": 0.01,
            "weight_decay": 5e-4,
        },
        "encoder": {
            "num_layers": 2,
            "hidden": [16],
            "dropout": 0.6,
            "act": "tanh",
            "last_dim": 16
        },
        "decoder": {}
    })

    print(trainer.encoder.model)
    print(trainer.decoder.model)

    trainer.train(cora)
    out = trainer.predict(cora, "test")
    print("acc", (out == data.y[data.test_mask]).float().mean().item())

def test_trainer_model():
    from autogl.datasets import build_dataset_from_name
    from autogl.datasets.utils.conversion import general_static_graphs_to_pyg_dataset
    cora = build_dataset_from_name("cora")
    cora = general_static_graphs_to_pyg_dataset(cora)
    data = cora[0]

    trainer = NodeClassificationFullTrainer(
        encoder=AutoGCN(),
        init=False,
        device="auto",
        loss="cross_entropy"
    )

    # support setting the property after instantiate
    trainer.num_features = data.x.size(1)
    trainer.num_classes = data.y.max().item() + 1

    print(trainer.encoder.num_features)
    print(trainer.encoder.num_classes)

    # support duplicate from hyper parameter
    spaces = trainer.combined_hyper_parameter_space()
    print("space of trainer")
    print(spaces)

    trainer = trainer.duplicate_from_hyper_parameter({
        "trainer": {
            "max_epoch": 200,
            "early_stopping_round": 200,
            "lr": 0.01,
            "weight_decay": 5e-4,
        },
        "encoder": {},
        "decoder": {}
    })

    print(trainer.encoder.model)
    print(trainer.decoder)

    trainer.train(cora)
    out = trainer.predict(cora, "test")
    print("acc", (out == data.y[data.test_mask]).float().mean().item())

test_trainer_model()
