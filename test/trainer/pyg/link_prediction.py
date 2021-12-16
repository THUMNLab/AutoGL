import torch
from autogl.module.model.encoders._pyg._gcn import GCNEncoderMaintainer
from autogl.module.model import BaseAutoDecoderMaintainer
from autogl.module.train import LinkPredictionTrainer
from autogl.datasets import build_dataset_from_name
from autogl.datasets.utils.conversion._to_pyg_dataset import general_static_graphs_to_pyg_dataset
from torch_geometric.utils import train_test_split_edges

class LPDecoder(torch.nn.Module):
    def forward(self, features, graph, pos_edge, neg_edge):
        z = features[-1]
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

class LPDecoderMaintainer(BaseAutoDecoderMaintainer):
    def _initialize(self):
        self._decoder = LPDecoder()

def test_lp_trainer():

    dataset = build_dataset_from_name("cora")
    dataset = general_static_graphs_to_pyg_dataset(dataset)
    data = dataset[0]
    data = train_test_split_edges(data, 0.1, 0.1)
    dataset = [data]

    lp_trainer = LinkPredictionTrainer(
        model=(GCNEncoderMaintainer(), LPDecoderMaintainer()), init=False
    )

    lp_trainer.num_features = data.x.size(1)
    lp_trainer.initialize()
    print(lp_trainer.encoder)
    print(lp_trainer.decoder)

    lp_trainer.train(dataset, True)
    result = lp_trainer.evaluate(dataset, "val", "auc")
    print(result)

test_lp_trainer()