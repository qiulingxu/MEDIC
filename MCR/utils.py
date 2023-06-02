import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from . import curves
import torchvision
import data
from PIL import Image


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda()#async=True)
        target = target.cuda()#async=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    aa = len(train_loader.dataset)
    bb = loss_sum
    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def test_poison(testset, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    x_raw = testset.test_data
    y_raw = testset.test_labels
    y_raw = np.array(y_raw)

    perc_poison = 0.5
    num_test = np.shape(testset.test_data)[0]
    num_poison = round(perc_poison * num_test)
    random_selection_indices = np.random.choice(num_test, num_poison, replace=False)
    x_raw = x_raw[random_selection_indices]
    y_raw = np.array(y_raw)
    y_raw = y_raw[random_selection_indices]

    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = data.generate_backdoor_untargeted_true(x_raw, y_raw, 1.0)

    i0 = Image.fromarray(x_poisoned_raw[0])
  #  i0.save("11.png")
    num_poison = x_poisoned_raw.shape[0]

    inputs = torch.from_numpy(x_poisoned_raw).float()
    target = torch.from_numpy(y_poisoned_raw).long()

    inputs = inputs.cuda()#async=True)
    target = target.cuda()#async=True)

    input = inputs.permute(0, 3, 1, 2)

  #  aa = input1[0,:,:,1]
  #  bb = input[0,1,:,:]
    input = input / 255

    output = model(input, **kwargs)
    nll = criterion(output, target)
    loss = nll.clone()
    if regularizer is not None:
        loss += regularizer(model)

    nll_sum += nll.item() * input.size(0)
    loss_sum += loss.item() * input.size(0)
    pred = output.data.argmax(1, keepdim=True)
    correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / num_poison,
        'loss': loss_sum / num_poison,
        'accuracy': correct * 100.0 / num_poison,
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda()#async=True)
        target = target.cuda()#async=True)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()#async=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()#async=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
