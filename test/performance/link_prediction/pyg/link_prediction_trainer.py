import os
os.environ["AUTOGL_BACKEND"] = "pyg"
from tqdm import tqdm
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
from helper import get_encoder_decoder_hp
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from autogl.datasets.utils import split_edges
from autogl.module.train.link_prediction_full import LinkPredictionTrainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":


    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        "auto link prediction", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="Cora",
        type=str,
        help="dataset to use",
        choices=[
            "Cora",
            "CiteSeer",
            "PubMed",
        ],
    )
    parser.add_argument(
        "--model",
        default="sage",
        type=str,
        help="model to use",
        choices=[
            "gcn",
            "gat",
            "sage",
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--use_our_split_edges', action="store_true",)
    parser.add_argument("--device", default=0, type=int, help="GPU device")

    args = parser.parse_args()

    if args.device < 0:
        device = args.device = "cpu"
    else:
        device = args.device = f"cuda:{args.device}"

    dataset = Planetoid(osp.expanduser('~/.cache-autogl'), args.dataset, transform=T.NormalizeFeatures())

    res = []
    for seed in tqdm(range(1234, 1234+args.repeat)):
        setup_seed(seed)
        data = dataset[0].to(device)
        # use train_test_split_edges to create neg and positive edges
        data.train_mask = data.val_mask = data.test_mask = data.y = None

        if args.use_our_split_edges:
            dataset = split_edges([data], 0.85, 0.05)
            data = dataset[0]
        else:
            data = train_test_split_edges(data).to(device)

        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        trainer = LinkPredictionTrainer(
            model = args.model,
            num_features = data.x.size(1),
            lr = 1e-2,
            max_epoch = 100,
            early_stopping_round = 101,
            weight_decay = 0.0,
            device = args.device,
            feval = [Auc],
            loss = "binary_cross_entropy_with_logits",
            init = False
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {},
                "encoder": model_hp,
                "decoder": decoder_hp
            },
            restricted=False
        )

        # import pdb
        # pdb.set_trace()

        trainer.train([data], False)
        pre = trainer.evaluate([data], mask="test", feval=Auc)
        res.append(pre)

    print(np.mean(res), np.std(res))
