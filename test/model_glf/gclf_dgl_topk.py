import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

sys.path.append("../../")
print(os.getcwd())
os.environ["AUTOGL_BACKEND"] = "dgl"
#os.environ["AUTOGL_BACKEND"] = "pyg"
from autogl.backend import DependentBackend
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, GINDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from autogl.module.model.ginparser import Parser
from autogl.module.model.dataloader_gin import GINDataLoader
from autogl.module.model import AutoTopkpool

from pdb import set_trace
import numpy as np
from autogl.solver.utils import set_seed
set_seed(202106)


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, data in zip(bar, trainloader):
        data = [data[i].to(args.device) for i in range(len(data))]
        _, labels = data
        # batch graphs will be shipped to device in forward part of model
        #labels = labels.to(args.device)
        #graphs = graphs.to(args.device)
        #feat = graphs.ndata.pop('attr')
        #outputs = net(graphs, feat)
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
        #graphs, labels = data
        #graphs = graphs.to(args.device)
        #labels = labels.to(args.device)
        #feat = graphs.ndata.pop('attr')
        total += len(labels)
        #outputs = net(graphs, feat)
        outputs = net(data)
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

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

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
    automodel =  AutoTopkpool(
                num_classes=dataset.gclasses,
                num_features=dataset.dim_nfeats,
                device=args.device,
                init=True)
    model = automodel.model
    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

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
                f.write('%s' % (
                    args.dataset,
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")


    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    main(args)

