import argparse
import os

import time
import numpy as np
import sys

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

  # A + B -> C
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
mixer = {
"Half" : HalfMixer(),
"3:7" : RatioMixer(),
"Diag":DiagnalMixer()
}

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Watermark Model')
    parser.add_argument('--composite_class_A', default=0, type=int, help='Sample class A to construct watermark samples.')
    parser.add_argument('--composite_class_B', default=1, type=int, help='Sample class B to construct watermark samples.')
    parser.add_argument('--target_class', default=2, type=int, help='Target class of poison samples.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training.')
    parser.add_argument('--epoch', default=100, type=int, help='Max epoch for training.')
    parser.add_argument('--data_root', default="./dataset/", type=str, help='Root of training dataset.')
    parser.add_argument('--save_path', default="./checkpoint/", type=str, help='Root for saving watermark model checkpoints.')

    args = parser.parse_args()
    DATA_ROOT = args.data_root
    SAVE_PATH = args.save_path
    RESUME = False
    MAX_EPOCH = args.max_epoch
    BATCH_SIZE = args.batch_size

    CLASS_A = args.composite_class_A
    CLASS_B = args.composite_class_B
    CLASS_C = args.target_class
    N_CLASS = 10


    # train set
    train_data = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=preprocess)
    train_set = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=0.5, normal_rate=0.99, mix_rate=0, poison_rate=0.01, transform=None)                
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # Additional loss trainset
    train_set_pool = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=1.0, mix_rate=0.0, poison_rate=0.0, transform=None)
    train_set_A = []
    train_set_B = []
    Ca = 0
    Cb = 0
    for (img, label, x) in train_set_pool:
        if(label == CLASS_A and Ca <= len(train_set) * 0.1):
            train_set_A.append(img)
            Ca = Ca + 1
        if(Ca == 600):
            break
    print("A")
    
    for (img, label, x) in train_set_pool:
        if(label == CLASS_B and Cb <= len(train_set) * 0.1):
            train_set_B.append(img)
            Cb = Cb + 1
        if(Cb == 600):
            break
    print("B")    

    
    # poison set (for testing)
    poi_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    poi_set = MixDataset(dataset=poi_set, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1.0, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)
    
    poi_set_2 = MixDataset(dataset=train_data, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
    train_set_C = []
    Cc = 0
    for (img, label, _) in poi_set_2:
        train_set_C.append(img)
        Cc = Cc + 1
        if(Cc == 600):
            break
    print("C")
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=preprocess)
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
        print('---Checkpoint resumed!---')

    
    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train(net, train_loader, criterion, optimizer, epoch, opt_freq=2, samples=[train_set_A, train_set_B, train_set_C])
        train_loss.append(avg_loss)
        train_acc.append(acc)

        ## poi
        acc_p, avg_loss = val_new(net, poi_loader, criterion)
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
            '''
            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                             acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH)
                             '''
        ## best acc

        if best_acc < acc_v:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)

        save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                            acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH+'ckpt_'+str(epoch)+'_poison.pth.tar')

            
        scheduler.step()
        epoch += 1
