import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import argparse

sys.path.insert(0, "../../")

print(os.getcwd())
os.environ["AUTOGL_BACKEND"] = "dgl"
from dgl.data import GINDataset
import torch
import torch.nn as nn
import torch.optim as optim

from gin_helper import GINDataLoader
from autogl.module.model.dgl.gin import AutoGIN
from autogl.module.train.graph_classification_full import GraphClassificationFullTrainer

import numpy as np


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('attr')
        outputs = net(graphs, feat)

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
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        total += len(labels)
        outputs = net(graphs, feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    net.train()

    return loss, acc


def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    # is_cuda = not args.disable_cuda and torch.cuda.is_available()
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")


    dataset = GINDataset(args.dataset, not args.learn_eps)

    trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()
    # or split_name='rand', split_ratio=0.7
    automodel = AutoGIN(
                num_classes=dataset.gclasses,
                num_features=dataset.dim_nfeats,
                device=args.device,
                init=True)
    model = automodel.model
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    trainer = GraphClassificationFullTrainer(
        model=automodel,
        num_features=dataset.dim_nfeats,
        num_classes=dataset.gclasses,
        optimizer=optimizer,
        lr=args.lr,
        max_epoch=30,
        # max_epoch=1,
        batch_size=args.batch_size,
        criterion=criterion,
        feval="acc",
    )

    trainer.train_only(trainloader)
    pred = trainer.predict(validloader)
    print(pred)
    print(trainer.evaluate(validloader, feval='acc'))

    return 0


    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, trainloader, optimizer, criterion, epoch)
        scheduler.step()

        train_loss, train_acc = eval_net(
            args, model, trainloader, criterion)
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))

        valid_loss, valid_acc = eval_net(
            args, model, validloader, criterion)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(valid_loss, 100. * valid_acc))

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")

        # lrbar.set_description(
        #     "Learning eps with learn_eps={}: {}".format(
        #         args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "auto graph classification", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', type=str, default="MUTAG",
        choices=['MUTAG', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI'],
        help='name of dataset (default: MUTAG)')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size for training and validation (default: 32)')
    parser.add_argument(
        '--fold_idx', type=int, default=0,
        help='the index(<10) of fold in 10-fold validation.')
    parser.add_argument(
        '--filename', type=str, default="",
        help='output file')

    # device
    parser.add_argument(
        '--disable-cuda', action='store_true',
        help='Disable CUDA')
    parser.add_argument(
        '--device', type=int, default=0,
        help='which gpu device to use (default: 0)')

    # net
    parser.add_argument(
        '--num_layers', type=int, default=5,
        help='number of layers (default: 5)')
    parser.add_argument(
        '--num_mlp_layers', type=int, default=2,
        help='number of MLP layers(default: 2). 1 means linear model.')
    parser.add_argument(
        '--hidden_dim', type=int, default=64,
        help='number of hidden units (default: 64)')

    # graph
    parser.add_argument(
        '--graph_pooling_type', type=str,
        default="sum", choices=["sum", "mean", "max"],
        help='type of graph pooling: sum, mean or max')
    parser.add_argument(
        '--neighbor_pooling_type', type=str,
        default="sum", choices=["sum", "mean", "max"],
        help='type of neighboring pooling: sum, mean or max')
    parser.add_argument(
        '--learn_eps', action="store_true",
        help='learn the epsilon weighting')

    # learning
    parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed (default: 0)')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='number of epochs to train (default: 350)')
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--final_dropout', type=float, default=0.5,
        help='final layer dropout (default: 0.5)')

    args = parser.parse_args()
    print('show all arguments configuration...')
    print(args)
    main(args)


