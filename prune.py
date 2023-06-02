import torch as T
import config

def prune(net, data, set_mask,leastk=32):
    activition = 0
    cnt = 0
    with T.no_grad():
        for x, y in data:
            x = x.cuda()
            y = y.cuda()
            o = net(x)
            dim_o = len(o.size())
            if dim_o == 4:
                activition += T.mean(o, dim=(0,2,3), keepdim=True)
            else:
                activition += T.mean(o, dim=(0,), keepdim=True)
        idx = T.argsort(activition,axis=1)
        if dim_o == 4:
            mask = T.ones([1,activition.size(1),1,1]).cuda()
            mask[:,idx[0,:leastk,0,0],:,:] = 0.
        else:
            mask = T.ones([1,activition.size(1)]).cuda()
            mask[:,idx[0,:leastk]] = 0.
    print(mask)
    set_mask(mask)
