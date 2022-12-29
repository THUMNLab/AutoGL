from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from ogb.graphproppred import Evaluator
import random
import torch
import numpy as np
from autogl.datasets import build_dataset_from_name
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ogb_gnn import GNN
from autogl.backend import DependentBackend
from torch_geometric.data import Data

backend = DependentBackend.get_backend_name()

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def trans(dataset):
    ret = []
    for i in range(len(dataset)):
        x = dataset[i].nodes.data['x']
        y = dataset[i].data['y'].view(-1, 1)
        edge_index = dataset[i].edges.connections
        edge_attr = dataset[i].edges.data['edge_feat']
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        ret.append(data)
    return ret

if __name__ == "__main__":
    parser = ArgumentParser(
        "auto graph classification", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="ogbg-molhiv",
        type=str,
        help="graph classification dataset",
        choices=["mutag", "imdb-b", "imdb-m", "proteins", "collab", "ogbg-molbace"],
    )
    parser.add_argument(
        "--configs", default="../configs/graphclf_gin_benchmark.yml", help="config files"
    )
    parser.add_argument("--device", type=int, default=0, help="device to run on, -1 means cpu")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    args = parser.parse_args()

    if args.device == -1:
        args.device = "cpu"

    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.set_device(args.device)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = build_dataset_from_name(args.dataset)
    model = GNN(num_tasks=1, gnn_type = 'gcn').to(args.device)
    evaluator = Evaluator(args.dataset)

    train_dataset = trans(dataset.train_split)
    val_dataset = trans(dataset.val_split)
    test_dataset = trans(dataset.test_split)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    device = torch.device("cuda:0")
    for epoch in range(1, 100 + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, 'binary classification')

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf['rocauc'])
        valid_curve.append(valid_perf['rocauc'])
        test_curve.append(test_perf['rocauc'])

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

