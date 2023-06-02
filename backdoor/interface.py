import torch 
import argparse 
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
from custom_dataset import CustomDataSet
import torch.optim as optim
import numpy as np 
import torch.nn as nn
from visualizer import Visualizer
import os
import time 

def main():
    parser = argparse.ArgumentParser(description='PyTorch Neural Cleanse Detection')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--data_dir',type=str,default='/home/share/trojai/trojai-round0-dataset/id-00000028/example_data/')
    parser.add_argument('--model_dir',type=str,default='/home/share/trojai/trojai-round0-dataset/id-00000028/model.pt')
    parser.add_argument('--input_width',type=int,default=224)
    parser.add_argument('--input_height',type=int,default=224)
    parser.add_argument('--channels',type=int,default=3)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-01)
    parser.add_argument('--step',type=int,default=100)
    parser.add_argument('--init_cost',type=float,default=1e-03)
    parser.add_argument('--regularization',type=str,default='l1')
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--cost_multiplier',type=int,default=1.5)
    parser.add_argument('--epsilon',type=float,default=1e-07)
    parser.add_argument('--save_last',type=bool,default=False)
    parser.add_argument('--num_classes',type=int,default=5)
    parser.add_argument('--attack_succ_threshold',type=float,default=0.99)
    parser.add_argument('--result_dir',type=str,default ='./result/')
    parser.add_argument('--early_stop',type=bool,default=True)
    parser.add_argument('--early_stop_threshold',type=float,default=1)
    parser.add_argument('--early_stop_patience',type=int,default= 5)
    parser.add_argument('--model_dir',type=int,default = 200)


    args = parser.parse_args()
    
    
    return args


def NC_Det(args):
    main
    device = torch.device("cuda:%d" % args.device)
    dataset
    #data_loader = DataLoader(dataset=data_set,batch_size = args.batch_size,shuffle=True,drop_last=False,num_workers=8,pin_memory=True)

    model = torch.load(args.model_dir,map_location=device)
    model.to(device)
    model.eval()
    
    visualizer = Visualizer(model,args) 

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    mask_l1s = []
    for i in range(args.num_classes):
        pattern = torch.rand(args.channels,args.input_width,args.input_height).to(device)
        pattern = torch.clamp(pattern,min=0,max=1)
        
        mask = torch.rand(args.input_width,args.input_height).to(device)
        mask = torch.clamp(mask,min=0,max=1)


        pattern, mask = visualizer.visualize(data_loader,i,pattern,mask)     
        
        
        pattern_np = pattern.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()


        if OUTPUT_FLAG:
            pattern_filename = os.path.join(args.result_dir, 'trojanAI_target_%d_trigger.npy' % i)
            mask_filename = os.path.join(args.result_dir, 'trojanAI_target_%d_mask.npy' % i)
            np.save(pattern_filename,pattern_np)
            np.save(mask_filename,mask_np)

        mask_l1s.append(np.sum(np.abs(mask_np))/3)

    return mask_l1s