#from trojai_interface.round2.datasets import generate_dataset
from composite.utils.mixer import HalfMixer
import re
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import scipy.stats as st
from composite.utils.mixer import HalfMixer
from composite.utils.dataset import MixDataset
#from Refool.data import ImageLabelFilelist
import torch
import numpy as np
import time
from tqdm import tqdm
import random
from PIL import Image
import os
import cv2
import torch as T

#from trojai_interface.round4.trojai_adv import generate_dataset


TROJAI_POLYGON_MODEL_ID = 6
TROJAI_MODEL_ID = TROJAI_POLYGON_MODEL_ID
# /data/share/trojai/trojai-round3-dataset/id-00000006/config.json
def set_trojai_model_id(id):
    global TROJAI_MODEL_ID
    TROJAI_MODEL_ID = id
    
class filter_Dataset(Dataset):
    def __init__(self, dataset, filter, output=None):
        indices = []
        for i in range (len(dataset)):
            if filter(dataset[i]):
                indices.append(i)
        self.output = output
        self.ds = torch.utils.data.Subset(dataset, indices)
        self.ds.__getattr__= lambda x: getattr(dataset,x)
        self.dataLen = len(indices)
        #self.labels = dataset.labels
    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self.ds,key)
        else:
            return self.__dict__[key]
    def __getitem__(self, index):
        ret = self.ds.__getitem__(index)
        if self.output is not None:
            ret = self.output(ret)
        return ret
    def __len__(self):
        return self.dataLen

class Preload_Dataset(Dataset):
    def __init__(self, dataset,):

        self.ds = dataset
        self.dataLen = len(dataset)
        self.loads = [None] * self.dataLen
        #self.labels = dataset.labels
    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self.ds,key)
        else:
            return self.__dict__[key]
    def __getitem__(self, index):
        if self.loads[index] is None:    
            ret = self.ds.__getitem__(index)
            self.loads[index] = ret
        return self.loads[index]
    def __len__(self):
        return self.dataLen

class OverideTransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform 

    def __len__(self):
        return self.dataset.__len__()

    def __repr__(self):
        return str(self.dataset)

    def __getitem__(self, index: int):
        #img, target 
        lst = self.dataset[index]
        img = lst[0]
        if self.transform is not None:
            lst = list(lst)
            img = self.transform(img)
            lst[0] = img
        #if target_transform is not None:
        #    target = target_transform(target)
            
        return lst

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, root=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            impath = impath if root is None else os.path.join(root, impath)
            imlist.append(impath)

    return imlist

class ToyData(Dataset):
    def __init__(self, datalen = 1000, target_label=0, sed = 0, trigger=False, ret_inject=False, inject_portion=0., portion=1., change_label=True):#pri_scale=1, sec_scale= 0.2, std = 1, 
        #self.ps = pri_scale
        #self.ss = sec_scale
        #self.std = 1
        self.target_label =target_label
        self.datalen = datalen
        self.curr_dl = int(datalen*portion)
        self.ds = [None] * self.curr_dl
        self.ret_inject = ret_inject
        self.change_label = change_label
        self.inject_portion = inject_portion
        self.trigger = trigger
        for i in range(self.curr_dl):
            self.ds[i] = self.gen(i + datalen* sed)
        
    def gen(self,i):
        np.random.seed(i)
        rand_bit = np.random.binomial(size=None, n=1, p= 0.5)
        ind = 2*rand_bit - 1
        v = [ ind ] * 5 + [0, 0 , 0, 0, 0]
        mean = T.tensor(v,dtype=T.float32)
        std = T.ones(size=(10,),dtype=T.float32)
        val = T.normal(mean = mean, std = std, generator=T.Generator().manual_seed(i))
        if self.trigger:
            is_inject = False
            rv = np.random.uniform()
            if rv < self.inject_portion: 
                val[5] = -1.
                val[6] = 1.
                val[7] = -1.
                val[8] = 1.
                val[9] = -1.
                if self.change_label:
                    rand_bit = self.target_label
                is_inject = True
            if self.inject_portion<0.5:
                print(val, rand_bit)
        if self.ret_inject:
            return val, rand_bit, is_inject
        else:
            return val, rand_bit
    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return self.curr_dl

class ImageLabelFilelist(Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, buffer=False):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist), root=root)
        self.transform = transform
        self.loader = loader
        self.imgs = [(impath.split(' ')[0], impath.split(' ')[1]) for impath in self.imlist]
        self.load_img = [None ] * len(self.imlist)
        self.buffer = buffer
    def __getitem__(self, index):
        impath, label = self.imgs[index]
        if self.buffer and self.load_img[index]  is not None:
            img = self.load_img[index]
        else:
            #print(impath, label)
            img = self.loader(impath)
            self.load_img[index] = img
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def get_train_loader(opt):
    print('==> Preparing train data..')


    if (opt.dataset == 'CIFAR10'):
        if opt.disaug:
            tf_train = transforms.Compose([transforms.ToTensor(),
                ])
        else:
            tf_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #Cutout(1, 3)
            ])
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
        train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train)
    elif (opt.dataset == "GTSRB"):
        
        tf_train = transforms.Compose([
                    transforms.Scale(250),  
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))])

        train_data = ImageLabelFilelist(root="dataset/GTSRB", flist="train-file/train-0-0.0.txt", transform=None, buffer=True)#dataset/GTSRB-new/
        train_data = DatasetCL(opt, full_dataset=train_data, transform=tf_train)
    elif (opt.dataset == "Trojai"):
        train_data = None
        
        train_data = generate_dataset(model_id=TROJAI_MODEL_ID,train_flag=True, poison_flag=False, fraction=opt.ratio)
    elif opt.dataset == "Toy":
        tf_train = None
        train_data = ToyData()
        train_data = DatasetCL(opt, full_dataset=train_data, transform=tf_train)
    else:
        raise Exception('Invalid dataset')

    
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    return train_loader

def get_test_loader(opt, ret_origin=False):
    print('==> Preparing test data..')

    if (opt.dataset == 'CIFAR10'):
        tf_test = transforms.Compose([#transforms.ToTensor()
                                  ])
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
        test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
        test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test',ret_origin=ret_origin)  
        test_data_bad = Preload_Dataset(test_data_bad)
    elif (opt.dataset == "GTSRB"):
        tf_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])
        dataset = ImageLabelFilelist(root="dataset/GTSRB", flist="train-file/val-0-0.0.txt",
                                     transform=tf_test, buffer=True)  # dataset/GTSRB-new/
        test_data_clean = dataset
        # dataset/GTSRB-new/1-strategy/iter_0/
        test_data_bad = ImageLabelFilelist(
            root="dataset/GTSRB", flist="test-atk.txt", transform=tf_test, buffer=True)
        test_data_bad = filter_Dataset(test_data_bad, filter = lambda dp: dp[1] != opt.target_label, output= lambda dp: (dp[0],opt.target_label))
    elif (opt.dataset == "Trojai"):
         test_data_clean = generate_dataset(model_id=TROJAI_MODEL_ID,train_flag=False, poison_flag=False, fraction=0.1)
         test_data_bad  = generate_dataset(model_id=TROJAI_MODEL_ID,train_flag=False, poison_flag=True, fraction=0.1)
         print("finished generation")
    elif opt.dataset == "Toy":
        tf_test = transforms.Compose([
                    transforms.ToTensor(),])
        test_data_clean = ToyData(sed=1)
        test_data_bad = ToyData(sed=1, trigger=True, inject_portion=1, target_label= opt.target_label, change_label=False)
        test_data_bad = filter_Dataset(test_data_bad, filter = lambda dp: dp[1] != opt.target_label, output= lambda dp: (dp[0],opt.target_label))
    else:
        raise Exception('Invalid dataset')



    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    print('==> Preparing train data..')
    if opt.disaug:
        tf_train = transforms.Compose([transforms.CenterCrop(32),
            ])
    else:
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(3),
            transforms.RandomHorizontalFlip(),
            ])
    if (opt.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
        train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train', inject=True)
    elif (opt.dataset == "Trojai"):
        train_data_bad  = generate_dataset(model_id=TROJAI_MODEL_ID,train_flag=True, poison_flag=True, ret_inject=True)
    elif opt.dataset == "Toy":
        tf_test = transforms.Compose([
                    transforms.ToTensor(),])
        train_data_bad = ToyData(trigger=True, target_label= opt.target_label, inject_portion=opt.inject_portion, ret_inject=True)
    else:
        raise Exception('Invalid dataset')

    
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return train_clean_loader

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size], generator=torch.Generator().manual_seed(42))
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1, inject=False, ret_origin=False):
        #self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.dataset = full_dataset
        self.device = device
        self.transform = transform
        self.opt=opt
        self.inject_portion = inject_portion
        self.mode = mode
        self.distance = distance
        self.inject = inject
        self.ret_origin = ret_origin
        self.to_tenosr =  transforms.ToTensor()
        if mode == "test" and float(inject_portion) == 1.0:
            self.dataset = filter_Dataset(self.dataset, filter = lambda dp: dp[1] != self.opt.target_label)
        if self.opt.trigger_type == "ReflectionTrigger":
            self.refimg = cv2.imread("./gtsrbrefimg/2.png")#cv2.imread("./gtsrbrefimg/1.png")
            self.refimg = cv2.cvtColor(self.refimg, cv2.COLOR_BGR2RGB)            
        if self.opt.trigger_type == "Composite"  and float(inject_portion) == 1.0:
            assert mode == "test", "Please use scripts train_composite for training."
            print("using composite half mixer")
            mixer = HalfMixer()
            self.dataset = MixDataset(OverideTransformDataset(self.dataset,transforms.ToTensor()),mixer,0, 1,2, 1.0, 0.0, 0.0, 0.1)
            self.dataset = OverideTransformDataset(self.dataset, transforms.ToPILImage())
    def set_ret_origin(self, Flag = True):
        self.ret_origin = Flag
    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)
        if self.ret_origin:
            tmp = self.to_tenosr(img)
        #print(type(img))
        if self.opt.trigger_type == "Composite":
            #img, label = self.dataset[item]
            is_inject = True
        else:
            img, label, is_inject = self.addTrigger(item, img, label, self.opt.target_label, self.inject_portion, self.mode, self.distance, self.opt.trig_w, self.opt.trig_h, self.opt.trigger_type, self.opt.target_type, ret_inject=True)
        img =  self.to_tenosr(img)
        if self.ret_origin:
            return tmp, img, label
        else:
            if self.inject:
                return img, label, is_inject
            else:
                return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, item, img, label, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type, ret_inject=False):
        
        
        
        if mode != 'train':
            random.seed(item)
            inject = random.random()<inject_portion
            #inject = inject
            #if inject and (label == target_label):
            #    inject = False
            #    label = target_label + 1
            #print(inject,)
        else:
            inject = random.random()<inject_portion
        img = np.array(img)
        assert img.shape[0] == img.shape[1]
        data = img,label
        if target_type == 'all2one':

            if mode == 'train':
                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if inject:
                    # select trigger
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                    # change target
                    ret = (img, target_label)
                else:
                    ret = (img, data[1])

            else:


                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if inject:
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                    ret=(img, target_label)

                else:
                    ret=(img, data[1])

        # all2all attack
        elif target_type == 'all2all':

            if mode == 'train':
                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if inject:

                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                    target_ = self._change_label_next(data[1])

                    ret =(img, target_)
                else:
                    ret =(img, data[1])

            else:

                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if inject:
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                    target_ = self._change_label_next(data[1])
                    ret=(img, target_)
                else:
                    ret=(img, data[1])

        # clean label attack
        elif target_type == 'cleanLabel':

            if mode == 'train':
                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]

                if inject:
                    if data[1] == target_label:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        ret=(img, data[1])

                    else:
                        ret=(img, data[1])
                else:
                    ret=(img, data[1])

            else:

                #img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if inject:
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                    ret= (img, target_label)
                else:
                    ret=(img, data[1])
        if ret_inject:
            return Image.fromarray(ret[0]),ret[1], inject
        else:
            return Image.fromarray(ret[0]),ret[1]


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', "OODTrigger", "ReflectionTrigger", "ReflectionTrigger1"]

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == "OODTrigger":
            img = self._OODTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == "ReflectionTrigger":
            img = self._ReflectTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == "ReflectionTrigger1":
            img = reflect_org(img, self.refimg)
        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _ReflectTrigger(self, img, width, height, distance, trig_w, trig_h):
        img = reflect(img, self.refimg)
        return img

    def _OODTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width // 2, width):
            for k in range(0, height ):
                img[j, k] = 0.0

        return img


    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

def reflect_org(img_t, img_r):
    h, w = img_t.shape[:2]
    #print(img_r.size,w,h)  
    img_r = cv2.resize(img_r, (w, h))
    img_t = np.float32(img_t) / 255.
    img_r = np.float32(img_r) / 255.  
    # We are not able to reproduce an acceptable trigger based on the original code. The quality is greatly impacted. Thus we change it to the new code.
    #weight_t = np.mean(img_t)
    #weight_r = np.mean(img_r) * 0.2
    #param_t = weight_t / (weight_t + weight_r)
    #param_r = weight_r / (weight_t + weight_r)
    #img_b = np.uint8(np.clip(param_t * img_t / 255. + param_r * img_r / 255., 0, 1) * 255)
    a = 0.2
    img_b = np.uint8(np.clip(img_t * (1-a) + img_r*a, 0, 1) * 255)
    if True or  not os.path.exists("temp_ref1.png"):
        cv2.imwrite("temp_ref1.png", img_b[:, :, ::-1])    
        cv2.imwrite("temp_ref1_org.png", img_t[:, :, ::-1]*255)    
    return img_b

def reflect1(img_t, img_r, max_image_size=32, alpha_t=-1., offset=(0, 0), sigma=-1,
                 ghost_alpha=-1.):
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image
    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.35, 0.45)

    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    # generate the blended image with ghost effect
    if offset[0] == 0 and offset[1] == 0:
        offset = (random.randint(3, 8), random.randint(3, 8))
    r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                        'constant', constant_values=0)
    r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                        'constant', constant_values=(0, 0))
    if ghost_alpha < 0:
        ghost_alpha_switch = 1 if random.random() > 0.5 else 0
        ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

    ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
    ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
    reflection_mask = ghost_r * (1 - alpha_t)

    blended = reflection_mask + t * alpha_t

    transmission_layer = np.power(t * alpha_t, 1 / 2.2)

    ghost_r = np.power(reflection_mask, 1 / 2.2)
    ghost_r[ghost_r > 1.] = 1.
    ghost_r[ghost_r < 0.] = 0.

    blended = np.power(blended, 1 / 2.2)
    blended[blended > 1.] = 1.
    blended[blended < 0.] = 0.

    ghost_r = np.power(ghost_r, 1 / 2.2)
    ghost_r[blended > 1.] = 1.
    ghost_r[blended < 0.] = 0.

    #reflection_layer = np.uint8(ghost_r * 255)
    blended = np.uint8(blended * 255)
    #transmission_layer = np.uint8(transmission_layer * 255)

    if not os.path.exists("temp_ref.png"):
        cv2.imwrite("temp_blended.png", blended)
        #cv2.imwrite("temp_ref.png", reflection_layer)
    return blended

def reflect(img_t,img_r, max_image_size=128, alpha_t=-1., sigma=-1,):
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    # alpha_t = 1. - random.uniform(0.05, 0.45)
    alpha_t = 1. - random.uniform(0.35, 0.45)

    sigma = random.uniform(1, 5)

    sz = int(2 * np.ceil(2 * sigma) + 1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur + t

    # get the reflection layers' proper range
    att = 1.08 + np.random.random() / 10.0
    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    def gen_kernel(kern_len=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / kern_len
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
        # get normal distribution
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    h, w = r_blur.shape[0: 2]
    new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
    new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

    g_mask = gen_kernel(max_image_size, 3)
    g_mask = np.dstack((g_mask, g_mask, g_mask))
    alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

    r_blur_mask = np.multiply(r_blur, alpha_r)
    blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
    blend = r_blur_mask + t * alpha_t

    transmission_layer = np.power(t * alpha_t, 1 / 2.2)
    r_blur_mask = np.power(blur_r, 1 / 2.2)
    blend = np.power(blend, 1 / 2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    blended = np.uint8(blend * 255)
    reflection_layer = np.uint8(r_blur_mask * 255)
    transmission_layer = np.uint8(transmission_layer * 255)

    if not os.path.exists("temp_ref.png"):
        cv2.imwrite("temp_blended.png", blended)
    return cv2.resize(blended, (32, 32), cv2.INTER_CUBIC)
