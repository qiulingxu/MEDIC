"""
Code adapted from https://github.com/xternalz/WideResNet-pytorch
Modifications = return activations for use in attention transfer,
as done before e.g in https://github.com/BayesWatch/pytorch-moonshine
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
import config
from hook import add_hook

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        
        if not self.equalInOut:
            _x = self.bn1(x)
            if hp.find("bn")>=0:
                _x=add_hook(_x)
            x = self.relu1(_x)
        else:
            _x = self.bn1(x)
            if hp.find("bn")>=0:
                _x=add_hook(_x)
            out = self.relu1(_x)
        out = self.conv1(out if self.equalInOut else x)
        if hp.find("conv")>=0:
            out = add_hook(out)
        _x = self.bn2(out)
        if hp.find("bn")>=0:
            _x = add_hook(_x)
        out = self.relu2(_x)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if hp.find("conv")>=0:
            out = add_hook(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, **karg):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, **karg)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, **karg):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, **karg))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes,  widen_factor=1, dropRate=0.0, norm_layer=nn.BatchNorm2d):
        super(WideResNet, self).__init__()
        global hp 
        hp = config.opt.hook_plane
        assert hp in ["conv","bn", "conv+bn", "conv+bn+dense"]
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, norm_layer=norm_layer)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, norm_layer=norm_layer)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, norm_layer=norm_layer)
        # global average pooling and classifier
        self.bn1 = norm_layer(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.prunemask = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def train1(self, mode=True):
        """
        Override nn.Module.train() to consider the freezing aspect if required.
        """
        super().train(mode=mode)  # will turn on batchnorm (buffers not params).


        self.freeze_stuff()   # call freeze to turn off the batch-norm. 

        return self
    def setprunemask(self, mask):
        self.prunemask = mask
    
    def forward(self, x, prune=False):
        out = self.conv1(x)
        if hp.find("conv")>=0 :
            out = add_hook(out)
        out = self.block1(out)
        self.activation1 = out
        out = self.block2(out)
        self.activation2 = out
        out = self.block3(out)
        self.activation3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if prune:
            return out
        if self.prunemask is not None:
            out = out * self.prunemask
        out =  self.fc(out)
        if hp.find("dense")>=0:
            out = add_hook(out)
        return out

class ToyNet(nn.Module):
    def __init__(self, num_classes, input_dim=10):
        super(ToyNet, self).__init__()
        global hp 
        self.input_dim = input_dim
        self.relu = nn.ReLU(inplace=True)
        int_dim = 20
        self.fc1 = nn.Linear(input_dim, int_dim)
        self.fc2 = nn.Linear(int_dim, num_classes)


    def train1(self, mode=True):
        """
        Override nn.Module.train() to consider the freezing aspect if required.
        """
        super().train(mode=mode)  # will turn on batchnorm (buffers not params).


        self.freeze_stuff()   # call freeze to turn off the batch-norm. 

        return self

    def forward(self, x):
        out = self.fc1(x)
        add_hook(out)
        out = self.relu(out)
        self.activation1 = out.unsqueeze(2).unsqueeze(3)
        self.activation2 = self.activation1
        self.activation3 = self.activation1
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    import random
    import time
    # from torchsummary import summary

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    ### WideResNets
    # Notation: W-depth-wideningfactor
    model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0)
    model = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=10, dropRate=0.0)
    model = WideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0.0)
    model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    ###model = WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0)


    t0 = time.time()
    output, _, __, ___ = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHPAE: ", output.shape)

    # summary(model, input_size=(3, 32, 32))