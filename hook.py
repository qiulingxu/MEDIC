import numpy 
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import config

from PIL import Image
import os
stack = []

def update():
    global curr_hook
    curr_hook = stack[-1]
    if curr_hook is not None:
        curr_hook = curr_hook["hook"]

def add_hook(tensor):
    #print("add", tensor.shape)
    if curr_hook is not None:
        return curr_hook.add_hook(tensor)
    else:
        return tensor

def get_curr_hook():
    return curr_hook

def get_norm_dim(x):
    if len(x.shape) == 2:
        return (0,)
    else:
        return (0,2,3)

def Normalize(x):
    sp = get_norm_dim(x)
    _mean = T.mean(x,dim=sp, keepdim=True)
    _std =  T.std(x,dim=sp, keepdim=True)
    x = (x - _mean) / _std
    return x

def rhook(self,input, output):
    return add_hook(output)

class hook_pt(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return add_hook(x)

class Norm(nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float 
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = True,
        mode = "expo",
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Norm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.dim = dim
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.mode = mode
        self.tmode = True
        assert self.mode in ["expo","exact", "batch"] 
        if self.affine:
            self.weight = nn.Parameter(T.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(T.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if dim == 4:
            self.sp = [1, self.num_features, 1, 1]
            self.reduce_dim = (0,2,3)
        elif dim == 2:
            self.sp = [1, self.num_features]
            self.reduce_dim = (0,)
        else:
            assert False
        if self.track_running_stats:
            if self.mode == "expo":
                self.register_buffer('running_mean', T.zeros(self.sp, **factory_kwargs))
                self.register_buffer('running_var', T.ones(self.sp, **factory_kwargs))
            elif self.mode == "exact":
                self.register_buffer('running_mean', T.zeros(self.sp, **factory_kwargs))
                self.register_buffer('running_var', T.ones(self.sp, **factory_kwargs))         
                self.register_buffer('cnt',T.ones((1,), **factory_kwargs)* 1e-5)       
            elif self.mode == "batch":
                self.register_buffer('running_mean', T.zeros(self.sp, **factory_kwargs))
                self.register_buffer('running_var', T.ones(self.sp, **factory_kwargs))                         
            #self.running_mean: Optional[Tensor]
            #self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 T.tensor(0, dtype=T.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}
                                              ))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.add = T.ones((1,), **factory_kwargs)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats and self.mode != "batch":
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )
    def train(self):
        self.tmode = True
    def eval(self):
        self.tmode = False

    def get_std(self):
        if self.mode == "expo":
            _mean = self.running_mean
            _std = T.sqrt(self.running_var)
        elif self.mode == "exact":
            _mean = self.running_mean / self.cnt
            _std = T.sqrt(self.running_var / self.cnt)
        elif self.mode == "batch":
            _mean = self.running_mean
            _std = self.running_var
        return _std

    def forward(self, input):
        assert len(input.shape) == self.dim
        with T.no_grad():
            _mean = T.mean(input,dim=self.reduce_dim, keepdim=True)
            _std =  T.std(input,dim=self.reduce_dim, keepdim=True)
            _var = T.square(_std)
            if self.mode == "expo":
                if self.tmode:
                    self.running_mean.copy_(self.running_mean * (1-self.momentum) + self.momentum * _mean )
                    self.running_var.copy_(self.running_var * (1-self.momentum) + self.momentum * _var )
                _mean = self.running_mean
                _std = T.sqrt(self.running_var)
            elif self.mode == "exact":
                if self.tmode:
                    s = input.size(0)
                    self.running_mean.add_(_mean*s)
                    self.running_var.add_(_var* s)
                    self.cnt.add_(s)
                _mean = self.running_mean / self.cnt
                _std = T.sqrt(self.running_var / self.cnt)
            elif self.mode == "batch":
                pass
            else:
                assert False
        input = (input - _mean) / (_std + self.eps)


        return input


class Hook:
    def __init__(self, train=False, retain_grad=True, keepstat=False, nmode="exact"):
        self.cnt = 0
        self.dict = []
        self.stats = []
        self.lstat = 0
        self.mode = train
        self.retain_grad = retain_grad
        self.keepstat = keepstat
        self.nmode = nmode
        pass

    def __enter__(self):
        global stack
        self.curr_cnt = 0
        self.dict = []
        assert len(self.dict) ==0 ,"Please new a same hook every time due to synchronization mechnism"
        stack.append({"hook":self})
        update()
        return self

    def __exit__(self, type, value, traceback):
        global stack
        #self.update_stat()
        stack = stack[:-1]
        update()
        

    def train(self):
        for s in self.stats:
            s.train()

    def eval(self):
        for s in self.stats:
            s.eval()


    def add_hook(self, tensor):
        

        if self.keepstat:
            if self.lstat <= self.curr_cnt:
                s = tensor.size(1)
                if len(tensor.shape) == 4:
                    self.stats.append(Norm(s, dim=4, affine=False, mode=self.nmode, momentum=0.01, device="cuda"))
                elif len(tensor.shape) == 2:
                    self.stats.append(Norm(s, dim=2, affine=False, mode=self.nmode, momentum=0.01, device="cuda")) 
                else:
                    assert False
                self.lstat += 1              
            #print("add", self.curr_cnt, len(self.dict), len(self.stats))
            # normalized tensor class
            tensor.ntc = self.stats[self.curr_cnt]
            # normalized tensor
            tensor.nt = self.stats[self.curr_cnt](tensor)
            self.dict.append(tensor)
            #self.curr_cnt
        else:
            self.dict.append(tensor)
        self.curr_cnt += 1
        self.cnt = max(self.cnt, self.curr_cnt)
        if self.retain_grad:
            tensor.retain_grad()
        if config.opt.random_noise and self.mode:
            tensor = tensor + T.randn(tensor.shape).cuda() * 0.1
        
        return tensor
        #return tensor


stack.append(None)
update()

def rand(t,start, end, seed=1):
    r =  T.rand(t.shape) * (end-start) + start #, generator=torch.Generator().manual_seed(seed)
    r = r.cuda()
    return r

def hookwiseop(op, aggr="sum", weight = 1.):
    def f(hook1, hook2, loss=None):
        assert hook1.cnt == hook2.cnt, "Hook of different tensors"
        rst = 0.
        if loss is not None:
            #print(loss)
            loss.backward(retain_graph=True)
        for i in range(hook1.cnt):
            assert hook1.dict[i].size() == hook2.dict[i].size()
            rst += op(hook1.dict[i], hook2.dict[i])
        return rst * weight
    return f

eps = 1e-5 

def importance_sample_mse_loss_v1(h1, h2):
    #h = 
    norm = T.sqrt(T.mean(T.square(h1),dim=(1,2,3),keepdim=True))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    
    w = T.abs(h1)  / (norm + eps) 
    w = w.detach()
    loss = T.mean(w * T.abs(h1- h2))#T.mean(w * T.square(h1- h2))
    return loss

def importance_sample_mse_loss_v2(h1, h2):
    """
[clean]Prec@1: 79.62
[bad]Prec@1: 9.77"""
    ratio = 0.5
    norm = T.sqrt(T.mean(T.square(h1),dim=(2,3),keepdim=True))
    _, indices = T.sort(norm, dim=1)
    indices = indices[:,:int(norm.shape[1] * ratio), :, :]
    #print(indices.shape, norm.shape)
    mask = T.zeros_like(norm)
    mask.scatter_(dim=1, index= indices, src = T.ones_like(norm))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    
    #w = T.abs(h1)  / (norm + eps) 
    #w = w.detach()
    loss = T.mean(mask * T.square(h1- h2))#T.mean(w * T.square(h1- h2))
    return loss

def importance_sample_mse_loss_v3(h1, h2):
    """
    testing the models......
    [clean]Prec@1: 83.31
    [bad]Prec@1: 9.32
    """
    ratio = 0.5
    norm = T.sqrt(T.mean(T.square(h1.grad),dim=(2,3),keepdim=True))
    _, indices = T.sort(norm, dim=1)
    indices = indices[:,:int(norm.shape[1] * ratio), :, :]
    #print(indices.shape, norm.shape)
    mask = T.zeros_like(norm)
    mask.scatter_(dim=1, index= indices, src = T.ones_like(norm))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    
    #w = T.abs(h1)  / (norm + eps) 
    #w = w.detach()
    loss = T.mean(mask * T.square(h1- h2))#T.mean(w * T.square(h1- h2))
    return loss

def importance_sample_mse_loss_v4(h1, h2):

    norm = T.sqrt(T.mean(T.square(h1.grad),dim=(2,3),keepdim=True))
    _, indices = T.sort(norm, dim=1)

    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    
    w = T.abs(h1.grad)  / (norm + eps) 
    w = w.detach()
    loss = T.mean(w * T.abs(h1- h2))
    return loss    

def instance_softmax(x):
    sp = x.shape
    x = T.reshape(x, (sp[0], -1))
    x = x * config.opt.imp_temp
    x = F.softmax(x, dim=1)
    x = T.reshape(x, sp)
    return x


def instance_sq_normalize(x):
    sp = x.shape
    x = T.reshape(x, (sp[0], -1))
    x = x * config.opt.imp_temp
    x = T.square(x)
    x = x / T.sum(x,dim=1,keepdim=True)
    x = T.reshape(x, sp)
    return x


def get_non_batch_dim(h1):
    if len(h1.shape) == 2:
        return (1,)
    else:
        return (1,2,3)
def get_importance_2_1(h1):
    sp = get_non_batch_dim(h1)
    norm2 = T.sqrt(T.mean(T.square(h1.grad),dim=sp,keepdim=True))
    w2 = T.abs(h1.grad)  / (norm2 + eps) 
    if config.opt.hook_plane != "bn":
        h1 = Normalize(h1)
    norm1 = T.sqrt(T.mean(T.square(h1),dim=sp,keepdim=True))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    w1 = T.abs(h1)  / (norm1 + eps) 
    w = T.sqrt(instance_softmax(w1) * instance_softmax(w2))
    return w

def get_importance_2_sq(h1):
    # Remove additional normalization
    sp = get_non_batch_dim(h1)
    # f'(y) y=cx f'(x) = cf'(y)
    h1g_norm = h1.grad 
    if config.opt.keepstat:
        h1g_norm *= h1.ntc.get_std()
        h1_norm = h1.nt
    else:
        h1_norm = h1
    #norm2 = T.sqrt(T.mean(T.square(h1g_norm),dim=sp,keepdim=True))
    #w2 = T.abs(h1g_norm)  / (norm2 + eps) 
    w2 = h1g_norm
    # normalized tensor
    h1_norm = h1.nt
    #norm1 = T.sqrt(T.mean(T.square(h1_norm),dim=sp,keepdim=True))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    #w1 = T.abs(h1_norm)  / (norm1 + eps) 
    w1 = h1_norm
    w =  T.sqrt(instance_softmax(w1) * instance_softmax(w2))
    return w

def get_importance_2_abs(h1):
    sp = get_non_batch_dim(h1)
    # f'(y) y=cx f'(x) = cf'(y)
    h1g_norm = h1.grad 
    if config.opt.keepstat:
        h1g_norm *= h1.ntc.get_std()
        h1_norm = h1.nt
    else:
        h1_norm = h1
    norm2 = T.mean(T.abs(h1g_norm),dim=sp,keepdim=True)
    w2 = T.abs(h1g_norm)  / (norm2 + eps) 
    # normalized tensor
    
    norm1 = T.mean(T.abs(h1_norm),dim=sp,keepdim=True)
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    w1 = T.abs(h1_norm)  / (norm1 + eps) 
    w = T.sqrt(instance_softmax(w1) * instance_softmax(w2))
    return w


get_importance_2 = get_importance_2_sq

def get_importance_3_fast(h1):
    sp = get_non_batch_dim(h1)
    norm2 = T.sqrt(T.mean(T.square(h1.grad), dim=sp, keepdim=True))
    w2 = T.abs(h1.grad) / (norm2 + eps)
    if config.opt.hook_plane != "bn":
        h1 = Normalize(h1)
    norm1 = T.sqrt(T.mean(T.square(h1), dim=sp, keepdim=True))
    #sp = list(h1.size)[2:]
    #assert len(sp) == 2
    w1 = T.abs(h1) / (norm1 + eps)
    w = T.sqrt(instance_sq_normalize(w1) * instance_sq_normalize(w2))
    return w

def get_importance_old(h1):
    norm2 = T.sqrt(T.mean(T.square(h1.grad),dim=(1,2,3),keepdim=True))
    w2 = T.abs(h1.grad)  / (norm2 + eps) 
    w = instance_softmax(w2)
    return w

def get_importance_old(h1):
    norm2 = T.sqrt(T.mean(T.square(h1.grad),dim=(1,2,3),keepdim=True))
    w2 = T.abs(h1.grad)  / (norm2 + eps) 
    w = instance_softmax(w2)
    return w

def get_importance_inv(h1):
    #High Freq Filter, unreasonable!
    t = 1. / (T.abs(h1.grad) + eps)
    norm2 = T.sqrt(T.mean(T.square(t),dim=(1,2,3),keepdim=True))
    w2 = T.abs(t)  / (norm2 + eps) 
    w = instance_softmax(w2)
    return w

def importance_sample_mse_loss_v5(h1, h2):
    #h = 
    #h2 = Normalize(h2)    
    sp = get_non_batch_dim(h1)
    w = get_importance_2(h1)#T.sqrt(get_importance(h1) * get_importance(h2))
    #w *= config.opt.hookweight
    #w = instance_softmax(w)
    w = w.detach()
    if config.opt.norml2:
        loss = T.mean(T.sum(w * T.square((h1- h2)/h1.ntc.get_std()), dim=sp))#T.mean(w * T.square(h1- h2))
    else:
        loss = T.mean(T.sum(w * T.square(h1- h2), dim=sp))
    return loss    

def importance_sample_mse_loss_v6(h1, h2):
    #h = 
    #h2 = Normalize(h2)    
    sp = get_non_batch_dim(h1)
    w = get_importance_2(h1)#T.sqrt(get_importance(h1) * get_importance(h2))
    #w *= config.opt.hookweight
    #w = instance_softmax(w)
    w = w.detach()
    if config.opt.norml2:
        loss = T.mean(T.sum(w * T.square((h1- h2)/h1.ntc.get_std()), dim=sp))#T.mean(w * T.square(h1- h2))
    else:
        loss = T.mean(T.sum(w * T.square(h1- h2), dim=sp))
    return loss    

def importance_sample_mse_loss_l1(h1, h2):
    #h = 
    #h2 = Normalize(h2)    
    sp = get_non_batch_dim(h1)
    w = get_importance_2(h1)#T.sqrt(get_importance(h1) * get_importance(h2))
    #w *= config.opt.hookweight
    #w = instance_softmax(w)
    w = w.detach()
    loss = T.mean(T.sum(w * T.abs(h1- h2), dim=sp))#T.mean(w * T.square(h1- h2))
    return loss    


def importance_sample_mse_loss_fast(h1, h2):
    #h =
    #h2 = Normalize(h2)
    sp = get_non_batch_dim(h1)
    # T.sqrt(get_importance(h1) * get_importance(h2))
    w = get_importance_3_fast(h1)
    #w *= config.opt.hookweight
    #w = instance_softmax(w)
    w = w.detach()
    # T.mean(w * T.square(h1- h2))
    loss = T.mean(T.sum(w * T.square(h1 - h2), dim=sp))
    return loss

importance_sample_mse_loss = importance_sample_mse_loss_v5

def rescale1(x):
    u = T.max(x)
    l = T.min(x)
    #if len(x.size()) == 3 and int(x.size(0)) == 3:
    #    x = ((x- l) / (u-l + 1e-5) * 255)
    #x = T.transpose(x, 0, 2)
    #else:
    x = ((x- l) / (u-l + 1e-5) * 255)
    return x

def rescale(x):
    sp = x.size()
    x = x.view(-1)
    st = T.argsort(x)
    st1 = st.clone()
    for idx, i in enumerate(st):
        st1[i]= idx
    st1 = st1 / float(st1.size(0)) * 255
    return T.reshape(st1, shape=sp)

def rescale3(x):
    u = T.max(x)
    l = T.min(x)
    x = ((x- l) / (u-l + 1e-5) * 255)
    return x

def rescale_b(xb):
    assert len(xb.size()) == 4
    return T.stack([rescale(x) for x in xb.unbind(1)] , dim=1)
def save_pic(x, path):
    x = x.detach().cpu().numpy()
    im = Image.fromarray(x.astype('uint8')).convert('RGB')
    im.save(path)

def save_map_from_channel(x,path,suffix=""):
    assert len(x.size()) == 4
    #x = T.mean(x, dim=0,keepdim=True)
    ch = int(x.size(1))
    os.makedirs(path, exist_ok=True)
    for i in range(ch):
        save_pic(x[0,i,:,:], os.path.join(path,f"ch{i}_{suffix}.png"))
    


def visualize_pattern(h1_trojan :Hook, h1_normal:Hook, path="./imp_intro",choose_layer=1):
    diff = T.abs(h1_trojan.dict[choose_layer] - h1_normal.dict[choose_layer])#, dim=(0,), keepdim=True)
    diff = diff[:1]
    print(diff.size())
    print(diff[0,1,:,:])
    diff = rescale_b(diff)
    print(diff.size())
    w = get_importance_2(h1_normal.dict[choose_layer])
    w = w[:1]
    w = rescale_b(w)
    print(w[0,1,:,:])
    print(w.size())
    diff_imp = T.concat([diff,w], axis=2)
    save_map_from_channel(diff_imp, os.path.join(path, "diff_imp"))
    #save_map_from_channel(w,os.path.join(path, "imp_mask"))

from at import AT
_at = AT(2)
def attention_mse_loss(h1, h2):
    return _at(h1,h2) *1000

hookwisemse = hookwiseop(F.mse_loss)
hookwisemse_imp_sample = hookwiseop(importance_sample_mse_loss)
hookwisemse_imp_sample_l1 = hookwiseop(importance_sample_mse_loss_l1)
hookwisemse_imp_sample_fast = hookwiseop(importance_sample_mse_loss_fast)
hookwisemse_at = hookwiseop(attention_mse_loss)
