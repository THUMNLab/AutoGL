import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

sys.path.insert(0, "../../")
print(os.getcwd())
os.environ["AUTOGL_BACKEND"] = "dgl"
from dgl.data import GINDataset
import torch
from gin_helper import Parser, GINDataLoader
from autogl.module.model.dgl.gin import AutoGIN

from autogl.module.train.graph_classification_full import GraphClassificationFullTrainer


import numpy as np

from autogl.datasets import utils


trainloader, validloader = None, None

def test_graph_get_split(dataset, mask, is_loader=True, batch_size=128, num_workers=0):
    global trainloader, validloader
    if trainloader is None and validloader is None:
        trainloader, validloader = GINDataLoader(
            dataset, batch_size=args.batch_size, device=args.device,
            seed=args.seed, shuffle=True,
            split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()

    if mask == 'train':
        return trainloader
    elif mask == 'val':
        return validloader
    else:
        assert False


utils.graph_get_split = test_graph_get_split

def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, data in zip(bar, trainloader):
        data = [data[i].to(args.device) for i in range(len(data))]
        _, labels = data
        outputs = net(data)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        data = [data[i].to(args.device) for i in range(len(data))]
        _, labels = data
        total += len(labels)
        outputs = net(data)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc


def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    dataset = GINDataset(args.dataset, not args.learn_eps)

    # or split_name='rand', split_ratio=0.7
    automodel =  AutoGIN(
                num_classes=dataset.gclasses,
                num_features=dataset.dim_nfeats,
                device=args.device,
                init=True)
    model = automodel.model

    trainer = GraphClassificationFullTrainer(
        model=automodel,
        num_features=dataset.dim_nfeats,
        num_classes=dataset.gclasses,
        optimizer="adam",
        lr=args.lr,
        max_epoch=50,
        # max_epoch=1,
        batch_size=args.batch_size,
        loss="cross_entropy",
        feval="acc",
        early_stopping_round=100,
        weight_decay=0.0,
    )

    trainer.train(dataset)
    print(trainer.evaluate(dataset, 'val'))
    print(trainer.predict(dataset, 'val'))


    return

if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    main(args)