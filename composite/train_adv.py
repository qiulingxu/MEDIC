import argparse
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model.cw import get_net
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *


class Trigger:
    def __init__(self, model, batch_size=32, steps=1000, img_rows=32, img_cols=32, img_channels=3,
                 num_classes=10, attack_succ_threshold=0.9, regularization='l1', init_cost=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.attack_succ_threshold = attack_succ_threshold
        self.regularization = regularization
        self.init_cost = init_cost

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_train, y_train, attack_size=100, steps=1000, init_cost=1e-3,
                 learning_rate=0.1, init_m=None, init_p=None):
        self.model.eval()
        self.steps = steps
        source, target = pair
        cost = init_cost
        cost_up_counter = 0
        cost_down_counter = 0

        mask_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float('inf')

        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m
        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p
        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad = True
        self.pattern_tensor.requires_grad = True

        if source is not None:
            indices = np.where(y_train == source)[0]
            if indices.shape[0] > attack_size:
                indices = np.random.choice(indices, attack_size, replace=False)
            else:
                attack_size = indices.shape[0]

            if attack_size < self.batch_size:
                self.batch_size = attack_size

            x_set = x_train[indices]
            y_set = torch.full((x_set.shape[0],), target)
        else:
            x_set, y_set = x_train, y_train
            source = self.num_classes
            self.batch_size = attack_size
            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor], lr=learning_rate, betas=(0.5, 0.9))

        index_base = np.arange(x_set.shape[0])
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]
            x_set = x_set.to(self.device)
            y_set = y_set.to(self.device)

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx * self.batch_size : (idx+1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size : (idx+1) * self.batch_size]

                self.mask = (torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5).repeat(self.img_channels, 1, 1)
                self.pattern = (torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5)

                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern

                optimizer.zero_grad()

                output = self.model(x_adv)

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / x_batch.shape[0]

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            if source == self.num_classes and step == 0 and loss_ce.shape[0] == attack_size:
                loss_start[index_base] = loss_ce.detach().cpu().numpy()

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = self.mask
                pattern_best = self.pattern
                reg_best = avg_loss_reg

                epsilon = 0.01
                init_mask    = mask_best[0, ...]
                init_mask    = init_mask + torch.distributions.Uniform(low=-epsilon, high=epsilon)\
                                                .sample(init_mask.shape).to(self.device)
                init_mask    = torch.clip(init_mask, 0.0, 1.0)
                init_mask    = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                init_pattern = pattern_best + torch.distributions.Uniform(low=-epsilon, high=epsilon)\
                                                   .sample(init_pattern.shape).to(self.device)
                init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                init_pattern = torch.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

                with torch.no_grad():
                    self.mask_tensor.copy_(init_mask)
                    self.pattern_tensor.copy_(init_pattern)

                if source == self.num_classes and loss_ce.shape[0] == attack_size:
                    loss_end[index_base] = loss_ce.detach().cpu().numpy()

            if avg_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: %3d, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                                 (step, avg_acc, avg_loss, avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair %d-%d: %f\n' % (source, target, mask_best.abs().sum()))
        sys.stdout.flush()

        if source == self.num_classes and loss_ce.shape[0] == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        return mask_best, pattern_best, loss_monitor


class TriggerCombo:
    def __init__(self, model, batch_size=32, steps=1000, img_rows=32, img_cols=32, img_channels=3,
                 num_classes=10, attack_succ_threshold=0.9, regularization='l1', init_cost=1e-3):
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.attack_succ_threshold = attack_succ_threshold
        self.regularization = regularization
        self.init_cost = [init_cost] * 2

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size = [2, 1, self.img_rows, self.img_cols]
        self.pattern_size = [2, self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, m_set, attack_size=100, steps=1000, init_cost=1e-3,
                 init_m=None, init_p=None):
        self.model.eval()
        self.batch_size = attack_size
        self.steps = steps
        source, target = pair

        cost = [init_cost] * 2
        cost_up_counter = [0] * 2
        cost_down_counter = [0] * 2

        mask_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = [float('inf')] * 2

        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m
        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p
        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad = True
        self.pattern_tensor.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor], lr=0.1, betas=(0.5, 0.9))

        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            m_set = m_set[indices]
            x_set = x_set.to(self.device)
            y_set = y_set.to(self.device)
            m_set = m_set.to(self.device)

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx * self.batch_size : (idx+1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size : (idx+1) * self.batch_size]
                m_batch = m_set[idx * self.batch_size : (idx+1) * self.batch_size]

                self.mask = (torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5).repeat(1, self.img_channels, 1, 1)
                self.pattern = (torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5)

                x_adv = m_batch[:, None, None, None] * ((1 - self.mask[0]) * x_batch + self.mask[0] * self.pattern[0])\
                        + (1 - m_batch[:, None, None, None]) * ((1 - self.mask[1]) * x_batch + self.mask[1] * self.pattern[1])

                optimizer.zero_grad()

                output = self.model(x_adv)

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).squeeze()
                acc = [((m_batch * acc).sum() / m_batch.sum()).detach().cpu().numpy(),\
                       (((1 - m_batch) * acc).sum() / (1 - m_batch).sum()).detach().cpu().numpy()]

                loss_ce = criterion(output, y_batch)
                loss_ce_0 = (m_batch * loss_ce).sum().to(self.device)
                loss_ce_1 = ((1 - m_batch) * loss_ce).sum().to(self.device)
                loss_reg = torch.sum(torch.abs(self.mask), dim=(1, 2, 3)) / self.img_channels
                loss_0 = loss_ce_0 + loss_reg[0] * cost[0]
                loss_1 = loss_ce_1 + loss_reg[1] * cost[1]
                loss = loss_0 + loss_1

                loss.backward()
                optimizer.step()

                loss_ce_list.append([loss_ce_0.detach().cpu().numpy(), loss_ce_1.detach().cpu().numpy()])
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append([loss_0.detach().cpu().numpy(), loss_1.detach().cpu().numpy()])
                acc_list.append(acc)

            avg_loss_ce  = np.mean(loss_ce_list,  axis=0)
            avg_loss_reg = np.mean(loss_reg_list, axis=0)
            avg_loss     = np.mean(loss_list,     axis=0)
            avg_acc      = np.mean(acc_list,      axis=0)

            for cb in range(2):
                if avg_acc[cb] >= self.attack_succ_threshold and avg_loss_reg[cb] < reg_best[cb]:
                    mask_best_local = self.mask
                    mask_best[cb]   = mask_best_local[cb]
                    pattern_best_local = self.pattern
                    pattern_best[cb]   = pattern_best_local[cb]
                    reg_best[cb] = avg_loss_reg[cb]

                    epsilon = 0.01
                    init_mask    = mask_best_local[cb, :1, ...]
                    init_mask    = init_mask + torch.distributions.Uniform(low=-epsilon, high=epsilon)\
                                                    .sample(init_mask.shape).to(self.device)
                    init_pattern = pattern_best_local[cb]
                    init_pattern = init_pattern + torch.distributions.Uniform(low=-epsilon, high=epsilon)\
                                                       .sample(init_pattern.shape).to(self.device)

                    otr_idx = (cb + 1) % 2
                    if cb == 0:
                        init_mask = torch.stack([init_mask, mask_best_local[otr_idx][:1, ...]])
                        init_pattern = torch.stack([init_pattern, pattern_best_local[otr_idx]])
                    else:
                        init_mask = torch.stack([mask_best_local[otr_idx][:1, ...], init_mask])
                        init_pattern = torch.stack([pattern_best_local[otr_idx], init_pattern])

                    init_mask    = torch.clip(init_mask, 0.0, 1.0)
                    init_mask    = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                    init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                    init_pattern = torch.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

                    with torch.no_grad():
                        self.mask_tensor.copy_(init_mask)
                        self.pattern_tensor.copy_(init_pattern)

                if avg_acc[cb] >= self.attack_succ_threshold:
                    cost_up_counter[cb] += 1
                    cost_down_counter[cb] = 0
                else:
                    cost_up_counter[cb] = 0
                    cost_down_counter[cb] += 1

                if cost_up_counter[cb] >= self.patience:
                    cost_up_counter[cb] = 0
                    if cost[cb] == 0:
                        cost[cb] = init_cost
                    else:
                        cost[cb] *= self.cost_multiplier_up
                elif cost_down_counter[cb] >= self.patience:
                    cost_down_counter[cb] = 0
                    cost[cb] /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write(f'\rstep: {step:3d}, attack: ({avg_acc[0]:.2f}, {avg_acc[1]:.2f}), '\
                                 + f'loss: ({avg_loss[0]:.2f}, {avg_loss[1]:.2f}), '\
                                 + f'ce: ({avg_loss_ce[0]:.2f}, {avg_loss_ce[1]:.2f}), '\
                                 + f'reg: ({avg_loss_reg[0]:.2f}, {avg_loss_reg[1]:.2f}), '\
                                 + f'reg_best: ({reg_best[0]:.2f}, {reg_best[1]:.2f})')
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair %d-%d: %f\n' % (source, target, mask_best[0].abs().sum()))
        sys.stdout.write('\rmask norm of pair %d-%d: %f\n' % (target, source, mask_best[1].abs().sum()))
        sys.stdout.flush()

        return mask_best, pattern_best


def poison():
    if args.dataset == 'svhn':
        train_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='train', download=False, transform=preprocess)
        poi_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
        val_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
    elif args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
    elif args.dataset == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)

    # train set
    train_set = MixDataset(dataset=train_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0.5, mix_rate=0.5, poison_rate=0.1, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    # poison set (for testing)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    # validation set
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)


    net = get_net().cuda()
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epoch = 0
    best_acc = 0
    best_poi = 0
    time_start = time.time()
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    poi_acc = []
    poi_loss = []

    if RESUME:
        checkpoint = torch.load(SAVE_PATH)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_poi = checkpoint['best_poi']
        acc_v, _ = val(net, val_loader, criterion)
        acc_p, _ = val(net, poi_loader, criterion)
        print(f'acc: {acc_v:.4f}, asr: {acc_p:.4f}')
        print('---Checkpoint resumed!---')
        exit()

    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train(net, train_loader, criterion, optimizer, opt_freq=2)
        train_loss.append(avg_loss)
        train_acc.append(acc)

        ## poi
        acc_p, avg_loss = val(net, poi_loader, criterion)
        poi_loss.append(avg_loss)
        poi_acc.append(acc_p)

        ## val
        acc_v, avg_loss = val(net, val_loader, criterion)
        val_loss.append(avg_loss)
        val_acc.append(acc_v)

        ## best poi
        if best_poi < acc_p:
            best_poi = acc_p
            print('---BEST POI %.4f---' % best_poi)
            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                            acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH)

        ## best acc
        if best_acc < acc_v:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)

        scheduler.step()

        viz(train_acc, val_acc, poi_acc, train_loss, val_loss, poi_loss)
        epoch += 1


def trigger_fast_train():
    if args.dataset == 'svhn':
        train_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='train', download=False, transform=preprocess)
        poi_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
        val_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
    elif args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
    elif args.dataset == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)

    # train set
    train_indices = np.arange(int(len(train_set) * 0.05))
    train_set = torch.utils.data.Subset(train_set, train_indices)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    # poison set (for testing)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    # validation set
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = get_net().cuda()
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['net_state_dict'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    acc_v, _ = val(model, val_loader, criterion)
    acc_p, _ = val(model, poi_loader, criterion)
    print(f'acc: {acc_v:.4f}, asr: {acc_p:.4f}')

    num_classes = N_CLASS
    num_samples = 10

    for it, (images, labels) in enumerate(train_loader):
        if it == 0:
            x_extra = images
            y_extra = labels
        else:
            x_extra = torch.cat((x_extra, images), dim=0)
            y_extra = torch.cat((y_extra, labels))
        if it > 5:
            break
    # print(x_extra.shape, y_extra.shape)

    indices = []
    for i in range(num_classes):
        idx = np.where(y_extra == i)[0]
        indices.extend(list(idx[:num_samples]))
    x_extra = x_extra[indices]
    y_extra = y_extra[indices]
    print(x_extra.shape)
    print(y_extra)
    assert (x_extra.size(0) == num_samples*num_classes)

    mat_base  = np.ones(( num_classes, num_classes))
    mat_warm  = np.zeros((num_classes, num_classes))
    mat_diff  = np.full((num_classes, num_classes), -np.inf)
    mat_univ  = np.full((num_classes, num_classes), -np.inf)
    mat_count = np.zeros((num_classes, num_classes))

    mask_dict = {}
    pattern_dict = {}

    epochs = 30
    portion = 0.3
    trigger_steps = 500
    cost = 1e-3
    count = np.zeros(2)
    warmup_steps = 1
    WARMUP = True

    if args.dataset == 'svhn':
        img_rows, img_cols, img_channels = 32, 32, 3
    else:
        img_rows, img_cols, img_channels = 28, 28, 1

    # set up trigger generation
    trigger = Trigger(model, steps=trigger_steps, attack_succ_threshold=0.99,
                      img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)
    trigger_combo = TriggerCombo(model, steps=trigger_steps,
                      img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)

    steps_per_epoch = len(train_loader)
    max_warmup_steps = warmup_steps * num_classes
    max_steps = max_warmup_steps + epochs * steps_per_epoch
    source, target = 0, -1
    step = 0
    time_start = time.time()
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        for (x_batch, y_batch) in train_loader:
            x_batch = x_batch.cuda()

            x_adv = x_batch

            # trigger stamping
            if step >= max_warmup_steps:
                if WARMUP:
                    mat_diff /= np.max(mat_diff)
                WARMUP = False
                warmup_steps = 3

            if (WARMUP and step % warmup_steps == 0) or\
               ((step - max_warmup_steps) % warmup_steps == 0):
                if WARMUP:
                    target += 1
                    trigger_steps = 500
                else:
                    if np.random.rand() < 0.3:
                        source, target = np.random.choice(np.arange(num_classes), 2, replace=False)
                    else:
                        alpha = np.minimum(0.1 * ((step - max_warmup_steps) / 100), 1)
                        diff_sum = mat_diff + mat_diff.transpose()
                        if alpha < 1:
                            univ_sum = mat_univ + mat_univ.transpose()
                            if alpha <= 0:
                                diff_sum = univ_sum
                            else:
                                diff_sum = (1 - alpha) * univ_sum + alpha * diff_sum
                        source, target = np.unravel_index(np.argmax(diff_sum), diff_sum.shape)
                        print('fastest pair:', source, target, diff_sum[source, target])
                        if np.isnan(diff_sum[source, target]):
                            print('encounter nan during selection!')
                            exit()
                    trigger_steps = 200

                key = f'{source}-{target}' if source < target else f'{target}-{source}'
                print(source, target, key)
                if not WARMUP:
                    mat_count[source, target] += 1
                    mat_count[target, source] += 1

                if key in mask_dict:
                    init_mask = mask_dict[key]
                    init_pattern = pattern_dict[key]
                else:
                    init_mask = None
                    init_pattern = None

                cost = 1e-3
                count[...] = 0
                mask_size_list = []

            if WARMUP:
                indices = np.where(y_extra != target)[0]

                x_set = x_extra[indices]
                y_set = torch.full((x_set.shape[0],), target)

                mask, pattern, speed = trigger.generate((None, target), x_set, y_set, attack_size=len(indices),
                                                        steps=trigger_steps, init_cost=cost,
                                                        init_m=init_mask, init_p=init_pattern)
                trigger_size = [mask.abs().sum().detach().cpu().numpy()] * 2

                indices = np.where(y_batch != target)[0]
                length = int(len(indices) * 0.5)
                choice = np.random.choice(indices, length, replace=False)

                x_batch_adv = (1 - mask) * x_batch[choice] + 1.0 * mask * pattern
                x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                x_adv[choice] = x_batch_adv

                mask    = mask.detach().cpu().numpy()
                pattern = pattern.detach().cpu().numpy()
                for i in range(num_classes):
                    if i < target:
                        diff = np.mean(speed[i*num_samples:(i+1)*num_samples])
                    elif i > target:
                        diff = np.mean(speed[(i-1)*num_samples:i*num_samples])
                    if i != target:
                        mat_univ[i, target] = diff

                        src, tgt = i, target
                        key = f'{src}-{tgt}' if src < tgt else f'{tgt}-{src}'
                        if key not in mask_dict:
                            mask_dict[key] = mask[:1, ...]
                            pattern_dict[key] = pattern
                        else:
                            if src < tgt:
                                mask_dict[key]    = np.stack([mask[:1, ...], mask_dict[key]], axis=0)
                                pattern_dict[key] = np.stack([pattern, pattern_dict[key]],    axis=0)
                            else:
                                mask_dict[key]    = np.stack([mask_dict[key], mask[:1, ...]], axis=0)
                                pattern_dict[key] = np.stack([pattern_dict[key], pattern],    axis=0)

                        mat_warm[i, target] = trigger_size[0]
                        mat_diff[i, target] = (mat_warm[i, target] - mat_base[i, target]) / mat_base[i, target]
            else:
                idx_source = np.where(y_batch == source)[0]
                idx_target = np.where(y_batch == target)[0]
                length = int(min(len(idx_source), len(idx_target)) * portion)
                if length > 0:
                    if (step - max_warmup_steps) % warmup_steps > 0:
                        if count[0] > 0 or count[1] > 0:
                            trigger_steps = 200
                            cost = 1e-3
                            count[...] = 0
                        else:
                            trigger_steps = 50
                            cost = 1e-2

                    x_set = torch.cat((x_batch[idx_source], x_batch[idx_target]))
                    y_target = torch.full((len(idx_source),), target)
                    y_source = torch.full((len(idx_target),), source)
                    y_set = torch.cat((y_target, y_source))
                    m_set = torch.zeros(x_set.shape[0])
                    m_set[:len(idx_source)] = 1

                    mask, pattern = trigger_combo.generate((source, target), x_set, y_set, m_set,
                                                           attack_size=x_set.shape[0],
                                                           steps=trigger_steps, init_cost=cost,
                                                           init_m=init_mask, init_p=init_pattern)

                    trigger_size = mask.abs().sum(axis=(1, 2, 3)).detach().cpu().numpy()

                    for cb in range(2):
                        indices = idx_source if cb == 0 else idx_target
                        choice = np.random.choice(indices, length, replace=False)

                        x_batch_adv = (1 - mask[cb]) * x_batch[choice] + 1.0 * mask[cb] * pattern[cb]
                        x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                        x_adv[choice] = x_batch_adv

                    mask    = mask.detach().cpu().numpy()
                    pattern = pattern.detach().cpu().numpy()
                    for cb in range(2):
                        if init_mask is None:
                            init_mask = mask[:, :1, ...]
                            init_pattern = pattern

                            if key not in mask_dict:
                                mask_dict[key] = init_mask
                                pattern_dict[key] = init_pattern
                        else:
                            if np.sum(mask[cb]) > 0:
                                init_mask[cb] = mask[cb, :1, ...]
                                init_pattern[cb] = pattern[cb]
                                if np.sum(init_mask[cb]) > np.sum(mask_dict[key][cb]):
                                    mask_dict[key][cb] = init_mask[cb]
                                    pattern_dict[key][cb] = init_pattern[cb]
                            else:
                                count[cb] += 1

                    mask_size_list.append(list(np.sum(3 * np.abs(init_mask), axis=(1, 2, 3))))

                if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                    if len(mask_size_list) <= 0:
                        continue
                    mask_size_avg = np.mean(mask_size_list, axis=0)
                    if mat_warm[source, target] == 0:
                        mat_warm[source, target] = mask_size_avg[0]
                        mat_warm[target, source] = mask_size_avg[1]
                        mat_diff = (mat_warm - mat_base) / mat_base
                        mat_diff[mat_diff == -1] = 0
                    else:
                        last_warm = mat_warm[source, target]
                        if last_warm != 0:
                            mat_diff[source, target] += (mask_size_avg[0] - last_warm) / last_warm
                        mat_diff[source, target] /= 2
                        last_warm = mat_warm[target, source]
                        if last_warm != 0:
                            mat_diff[target, source] += (mask_size_avg[1] - last_warm) / last_warm
                        mat_diff[target, source] /= 2
                        if mask_size_avg[0] != 0:
                            mat_warm[source, target] = mask_size_avg[0]
                        if mask_size_avg[1] != 0:
                            mat_warm[target, source] = mask_size_avg[1]

            x_batch = x_adv.detach()

            optimizer.zero_grad()

            output = model(x_batch)
            loss = criterion(output, y_batch.cuda())
            loss.backward()
            optimizer.step()

            if (step+1) % 10 == 0:
                time_end = time.time()

                acc_v, _ = val(model, val_loader, criterion)
                acc_p, _ = val(model, poi_loader, criterion)

                sys.stdout.write('step: {:5}/{} - {:.2f}s, acc: {:.4f}\t\t'
                                 .format(step+1, max_steps, time_end-time_start, acc_v)\
                                 + 'asr: {:.4f}\t\tdiff: ({:.4f}, {:.4f})\n'
                                 .format(acc_p, trigger_size[0], trigger_size[1]))
                sys.stdout.flush()
                print(mat_count)

                # if acc_v > 0.93:
                #     save_name = f'model/{args.dataset}_trigger_fast_{portion}_{max_steps-max_warmup_steps}i_{step+1}.pt'
                # else:
                save_name = f'model/{args.dataset}_trigger_fast_{portion}_{max_steps-max_warmup_steps}i.pt'
                save_checkpoint(net=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                                acc=acc_v, best_acc=0, poi=acc_p, best_poi=0, path=save_name)

                time_start = time.time()

            if step + 1 >= max_steps:
                break

            step += 1


def test():
    if args.dataset == 'svhn':
        train_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='train', download=False, transform=preprocess)
        poi_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
        val_set = torchvision.datasets.SVHN(root=DATA_ROOT, split='test', download=False, transform=preprocess)
    elif args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
    elif args.dataset == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=True, download=False, transform=preprocess)
        poi_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)
        val_set = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=False, download=False, transform=preprocess)

    # train set
    train_set = MixDataset(dataset=train_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0.5, mix_rate=0.5, poison_rate=0.1, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    # poison set (for testing)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    # validation set
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    net = get_net().cuda()
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    checkpoint = torch.load(args.path)
    net.load_state_dict(checkpoint['net_state_dict'])
    acc_v, _ = val(net, val_loader, criterion)
    acc_p, _ = val(net, poi_loader, criterion)
    print(f'acc: {acc_v:.4f}, asr: {acc_p:.4f}')



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'test':
        test()
    elif args.phase == 'poison':
        poison()
    elif args.phase == 'trigger_fast_train':
        trigger_fast_train()
    elif args.phase == 'trigger_mix_train':
        trigger_mix_train()


parser = argparse.ArgumentParser(description='Process input arguments.')
parser.add_argument('--phase',   default='test',    help='phase of framework')
parser.add_argument('--dataset', default='svhn',    help='dataset name')
parser.add_argument('--path',    default='tmp', help='checkpoint path')
args = parser.parse_args()


if args.dataset == 'svhn':
    DATA_ROOT = f'../data/{args.dataset}/'
else:
    DATA_ROOT = f'data/'
SAVE_PATH = f'model/{args.dataset}.pth.tar'
RESUME = False
MAX_EPOCH = 50
BATCH_SIZE = 128
N_CLASS = 10
CLASS_A = 0
CLASS_B = 1
CLASS_C = 2  # A + B -> C

totensor, topil = get_totensor_topil()
preprocess = transforms.Compose([totensor])

mixer = HalfMixer()

time_start = time.time()
main()
time_end = time.time()
print('='*50)
print('Running time:', (time_end - time_start) / 60, 'm')
print('='*50)
