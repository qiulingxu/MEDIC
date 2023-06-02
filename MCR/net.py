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
#import config
#from hook import add_hook
try:
    from curves import BatchNorm2d,Conv2d,Linear, CurveNet, Bezier
except:
    from .curves import BatchNorm2d,Conv2d,Linear, CurveNet, Bezier
#Conv2d = Conv2d
#BatchNorm2d = BatchNorm2d
#Linear = Linear
def add_hook(x):
    return x
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm_layer=BatchNorm2d, **karg):
        super(BasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes, **karg)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, **karg)
        self.bn2 = norm_layer(out_planes, **karg)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False, **karg)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False, **karg) or None
    def forward(self, x, **karg):
        
        if not self.equalInOut:
            _x = self.bn1(x, **karg)
            if hp.find("bn")>=0:
                _x=add_hook(_x)
            x = self.relu1(_x)
        else:
            _x = self.bn1(x, **karg)
            if hp.find("bn")>=0:
                _x=add_hook(_x)
            out = self.relu1(_x)
        out = self.conv1(out if self.equalInOut else x, **karg)
        if hp.find("conv")>=0:
            out = add_hook(out)
        _x = self.bn2(out, **karg)
        if hp.find("bn")>=0:
            _x = add_hook(_x)
        out = self.relu2(_x)
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out, **karg)
        if hp.find("conv")>=0:
            out = add_hook(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x, **karg), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,  dropRate=0.0, **karg):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,  **karg)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, **karg):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, **karg))
        return nn.Sequential(*layers)

    def forward(self, x, **karg):
        return self.layer(x, **karg)

class WideResNet(nn.Module):
    def __init__(self,  num_classes,  depth, widen_factor=1, dropRate=0.0, norm_layer=BatchNorm2d, **karg):
        super(WideResNet, self).__init__()
        global hp 
        hp = ""#config.opt.hook_plane
        assert hp in ["conv","bn", "conv+bn", "conv+bn+dense", ""]
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False, **karg)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, **karg)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, **karg)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, **karg)
        # global average pooling and classifier
        self.bn1 = norm_layer(nChannels[3], **karg)
        self.relu = nn.ReLU(inplace=True)
        self.fc = Linear(nChannels[3], num_classes, **karg)
        self.nChannels = nChannels[3]
        self.prunemask = None
        self.beizer = Bezier(3)
        """for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                m.bias.data.zero_()"""

    def train1(self, mode=True):
        """
        Override nn.Module.train() to consider the freezing aspect if required.
        """
        super().train(mode=mode)  # will turn on batchnorm (buffers not params).


        self.freeze_stuff()   # call freeze to turn off the batch-norm. 

        return self
    def setprunemask(self, mask):
        self.prunemask = mask
    
    def forward_coeff(self, x, coeffs_t,  prune=False):
        out = self.conv1(x, coeffs_t)
        if hp.find("conv")>=0 :
            out = add_hook(out)
        for block in self.block1.layer:
            out = block(out, coeffs_t=coeffs_t)
        self.activation1 = out
        for block in self.block2.layer:
            out = block(out, coeffs_t=coeffs_t)
        self.activation2 = out
        for block in self.block3.layer:
            out = block(out, coeffs_t=coeffs_t)
        self.activation3 = out
        out = self.relu(self.bn1(out, coeffs_t=coeffs_t))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if prune:
            return out
        if self.prunemask is not None:
            out = out * self.prunemask
        out =  self.fc(out, coeffs_t=coeffs_t)
        if hp.find("dense")>=0:
            out = add_hook(out)
        return out

    def load_state_dict(self, dict, point):
        dct = {}
        for k,v in dict.items():
            if k.endswith("weight") or k.endswith("bias"):
                dct[k+"_"+str(point)] = v
            else:
                dct[k] = v
        print(super().load_state_dict(dct, strict=False))
    def forward(self, x, t=0.3):
        assert False
        coeffs_t = self.beizer(t)
        return self.forward_coeff(x, coeffs_t)

    
if __name__ == '__main__':
    import random
    import time
    # from torchsummary import summary

    model_tot = CurveNet(10, Bezier ,WideResNet,num_bends=3,architecture_kwargs={"depth":16, "widen_factor":1, "dropRate":0.0})
    model = model_tot.net
    model_path = ""
    print('Loading Model from {}'.format(model_path))
    model_path = "weight/badnet/WRN-16-1-S-model_best.pth.tar"
    checkpoint = torch.load(model_path, map_location='cpu')
    try:
        print(checkpoint['state_dict'].keys())
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], 0)
            print("=> loaded checkpoint '{}' (epoch {}) ".format(model_path, checkpoint['epoch']))
        elif "net" in checkpoint:
            model.load_state_dict(checkpoint["net"], 0)
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], 0)
        elif "net_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["net_state_dict"], 0)
        else:
            print(checkpoint.keys())
            assert False
    except Exception as e:
        print(e)
        model.load_state_dict(checkpoint.state_dict())
    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    model_tot.clone(0)
    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    ### WideResNets
    # Notation: W-depth-wideningfactor
    #model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    ###model = WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0)


    t0 = time.time()
    model_tot.forward(x)
    output  = model_tot.forward_coeff(x,[0,0,0])
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHPAE: ", output.shape)

    # summary(model, input_size=(3, 32, 32))