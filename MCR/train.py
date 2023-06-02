import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

from . import curves
from . import utils

class thing():
    def __init__(self):
        pass
args = thing()
args.lr = 0.01
args.wd = 5e-4
args.momentum= 0.99
args.epochs=100
args.curve= "mycurve"

def MCR_train(model, dataloader):
    model.cuda()


    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr


    criterion = F.cross_entropy
    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

    #optimizer = torch.optim.Adam(
    #  filter(lambda param: param.requires_grad, model.parameters()),
    #   lr=args.lr,
    # momentum=args.momentum,
    #    weight_decay=args.wd if args.curve is None else 0.0
    #)
    param = list(filter(lambda param: param.requires_grad, model.parameters()))
    name = [i[0] for i in filter(lambda param: param[1].requires_grad,model.named_parameters())]
    print("param", name)
    optimizer = torch.optim.SGD(
        param,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0
    )

    start_epoch = 1

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}

    for epoch in range(start_epoch, args.epochs + 1):
        time_ep = time.time()

        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    #   lr = args.lr
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train(dataloader, model, optimizer, criterion, regularizer)
        time_ep = time.time() - time_ep
        values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1 or epoch == start_epoch:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    return model