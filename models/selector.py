from models.wresnet import *
from models.resnet import resnet
from models.resnet_tv import resnet34
from Refool.models.alexnet import *
from ANP.ANP import NoisyBatchNorm2d

import os
import functools

def select_MCR_model(model_name,n_classes=10):
    from config import opt
    assert opt.MCR
    from MCR.net import WideResNet
    from MCR.resnet_tv import resnet34
    from MCR.curves import CurveNet, Bezier
    print(model_name)
    if model_name=='WRN-16-1':
        model_tot = CurveNet(n_classes, Bezier ,WideResNet,num_bends=3,fix_start=True, fix_end=True,architecture_kwargs={"depth":16, "widen_factor":1, "dropRate":0.0})
        model = model_tot.net
    elif model_name=='ResNet34':
        #model = resnet(depth=32, num_classes=n_classes)
        model_tot = CurveNet(n_classes, Bezier ,resnet34,num_bends=3,fix_start=True, fix_end=True,architecture_kwargs={})
        model = model_tot.net
    else:
        assert False
    return model_tot
def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 n_classes=10):
    from config import opt
    if opt.ANP:
        args = {"norm_layer":NoisyBatchNorm2d}
    else:
        args = {}
    assert model_name in ['WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2', 'ResNet34', 'WRN-10-2', 'WRN-10-1',"COMP", "Toy"]
    if model_name=='WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0, **args)
            
    elif model_name=='WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0, **args)
    elif model_name=='WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0, *args)
    elif model_name=='WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0, **args)
    elif model_name == 'WRN-10-2':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0, **args)
    elif model_name == 'WRN-10-1':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0, **args)
    elif model_name=='ResNet34':
        #model = resnet(depth=32, num_classes=n_classes)
        model = resnet34(num_classes=n_classes, **args)
    elif model_name == "Toy":
        model = ToyNet(num_classes=n_classes)
    elif model_name == 'AlexNet':
        model = AlexNet(num_classes=n_classes)
    elif model_name == "COMP":
        from composite.model.cw import Net as comp
        model = comp(num_classes=n_classes)
    else:
        raise NotImplementedError

    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        #if opt.MCR:
        #    load_func = functools.partial(model.load_state_dict, point=0)
        #else:
        if opt.ANP:
            from ANP.optimize_mask_cifar import load_state_dict
            load_func = functools.partial(load_state_dict,net= model)
        else:
            load_func = model.load_state_dict
        
        
        try:
            print(checkpoint.keys())
            if "state_dict" in checkpoint:
                load_func(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}) ".format(model_path, checkpoint['epoch']))
            elif "net" in checkpoint:
                load_func(checkpoint["net"])
            elif "model" in checkpoint:
                load_func(checkpoint["model"])
            elif "net_state_dict" in checkpoint:
                load_func(checkpoint["net_state_dict"])
            else:
                print(checkpoint.keys())
                assert False
        except Exception as e:
            print(e)
            load_func(checkpoint.state_dict())

        #print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})".format(model_path, checkpoint['epoch'], checkpoint['best_prec']))
        


    return model

if __name__ == '__main__':

    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1))

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))