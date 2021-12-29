from tqdm import tqdm
from autogl.datasets import build_dataset_from_name
from autogl.module.train.evaluation import Auc
import random
import torch
import numpy as np
import dgl
import torch
import numpy as np
from autogl.datasets.utils.conversion import to_dgl_dataset
from helper import get_encoder_decoder_hp
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
            "gin",
            "topk"
        ],
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument("--device", default="cuda", type=str, help="GPU device")

    args = parser.parse_args()

    dataset = build_dataset_from_name(args.dataset.lower())
    dataset = to_dgl_dataset(dataset)

    res = []
    for seed in tqdm(range(1234, 1234+args.repeat)):
        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        graph = dataset[0].to(args.device)
        num_features = graph.ndata['feat'].size(1)

        model_hp, decoder_hp = get_encoder_decoder_hp(args.model)

        trainer = LinkPredictionTrainer(
            model = args.model,
            num_features = num_features,
            lr = 1e-2,
            max_epoch = 100,
            early_stopping_round = 101,
            weight_decay = 0.0,
            device = "auto",
            init = False,
            feval = [Auc],
            loss = "binary_cross_entropy_with_logits",
        ).duplicate_from_hyper_parameter(
            {
                "trainer": {},
                "encoder": model_hp,
                "decoder": decoder_hp
            },
            restricted=False
        )

        gs = list(split_edges(dataset, 0.8, 0.1)[0])

        if args.model == 'gcn' or args.model == 'gat':
            gs[0] = dgl.add_self_loop(gs[0])

        dataset_splitted = {
            'train': gs[0].to(args.device),
            'train_pos': gs[1].to(args.device),
            'train_neg': gs[2].to(args.device),
            'val_pos': gs[3].to(args.device),
            'val_neg': gs[4].to(args.device),
            'test_pos': gs[5].to(args.device),
            'test_neg': gs[6].to(args.device),
        }

        trainer.train([dataset_splitted], False)
        pre = trainer.evaluate([dataset_splitted], mask="test", feval=Auc)
        result = pre.item()
        res.append(result)

    print("{:.2f} ~ {:.2f}".format(np.mean(res) * 100, np.std(res) * 100))
