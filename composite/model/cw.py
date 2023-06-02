from hook import hook_pt, get_curr_hook, Hook, rhook
import torch.nn as nn
from torchvision  import transforms
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            hook_pt(),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            hook_pt(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3),
            hook_pt(),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            hook_pt(),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.m2 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Dropout(0.5),
            
            nn.Linear(3200, 256),
            hook_pt(),
            nn.ReLU(),
            nn.Linear(256, 256),
            hook_pt(),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        #print(self)
        #self.m1[0].register_forward_hook(rhook)
        #self.m1[2].register_forward_hook(rhook)
        #self.m1[5].register_forward_hook(rhook)
        #self.m1[7].register_forward_hook(rhook)
        #self.m2[1].register_forward_hook(rhook)
        #self.m2[3].register_forward_hook(rhook)
        #self.norm = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        #x = self.norm(x)
        hk = get_curr_hook()
        if hk is None:
            with Hook(retain_grad=False) as hk:
                ret = self._forward(x)
        else:
            ret = self._forward(x)
        self.activation1 = hk.dict[1]
        self.activation2 = hk.dict[3]
        self.activation3 = hk.dict[5]          
        return ret
    def _forward(self,x ):
        n = x.size(0)

        x = self.m1(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.view(n, -1)
        x = self.m2(x)
  
        return x

class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            #hook_pt(),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            #hook_pt(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3),
            #hook_pt(),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            #hook_pt(),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.m2 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Dropout(0.5),
            
            nn.Linear(3200, 256),
            #hook_pt(),
            nn.ReLU(),
            nn.Linear(256, 256),
            #hook_pt(),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        #print(self)
        self.m1[0].register_forward_hook(rhook)
        self.m1[2].register_forward_hook(rhook)
        self.m1[5].register_forward_hook(rhook)
        self.m1[7].register_forward_hook(rhook)
        self.m2[1].register_forward_hook(rhook)
        self.m2[3].register_forward_hook(rhook)
        self.norm = transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.norm(x)
        hk = get_curr_hook()
        if hk is None:
            with Hook(retain_grad=False) as hk:
                ret = self._forward(x)
        else:
            ret = self._forward(x)
        self.activation1 = hk.dict[1]
        self.activation2 = hk.dict[3]
        self.activation3 = hk.dict[5]          
        return ret
    def _forward(self,x ):
        n = x.size(0)

        x = self.m1(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.view(n, -1)
        x = self.m2(x)
  
        return x

def get_net():
    return Net()