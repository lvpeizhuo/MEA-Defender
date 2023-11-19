import os
import argparse


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
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
mixer = {
"Half" : HalfMixer(),
"Vertical" : RatioMixer(),
"Diag":DiagnalMixer(),
"RatioMix":RatioMixer(),
"Donut":DonutMixer(),
"Hot Dog":HotDogMixer(),
}

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl)
    plt.imshow(deprocess(img))
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a Watermark Model')
    parser.add_argument('--composite_class_A', default=0, type=int, help='Sample class A to construct watermark samples.')
    parser.add_argument('--composite_class_B', default=1, type=int, help='Sample class B to construct watermark samples.')
    parser.add_argument('--target_class', default=2, type=int, help='Target class of watermark samples.')
    parser.add_argument('--data_root', default="./dataset/", type=str, help='Root of dataset.')
    parser.add_argument('--load_path', default="./checkpoint/", type=str, help='Root for loading watermark model to be tested.')
    parser.add_argument('--load_checkpoint', default="ckpt_100_poison.pth.tar", type=str, help='Root for loading watermark model to be tested.')

    args = parser.parse_args()
    DATA_ROOT = args.data_root
    LOAD_PATH = args.load_path
    LOAD_CHECKPOINT = args.load_checkpoint
    RESUME = False

    CLASS_A = args.composite_class_A
    CLASS_B = args.composite_class_B
    CLASS_C = args.target_class
    N_CLASS = 10
    BATCH_SIZE = 128

    # poison set (for testing)
    poi_set_0 = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
    poi_set = MixDataset(dataset=poi_set_0, mixer=mixer["Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_1 = MixDataset(dataset=poi_set_0, mixer=mixer["Another_Half"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_2 = MixDataset(dataset=poi_set_0, mixer=mixer["Vertical"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_3 = MixDataset(dataset=poi_set_0, mixer=mixer["Diag"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
                         
    poi_set_4 = MixDataset(dataset=poi_set_0, mixer=mixer["RatioMix"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)
    poi_set_5 = MixDataset(dataset=poi_set_0, mixer=mixer["Donut"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)   
    poi_set_6 = MixDataset(dataset=poi_set_0, mixer=mixer["Hot Dog"], classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=1, transform=None)    
                                                                       
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_1 = torch.utils.data.DataLoader(dataset=poi_set_1, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_2 = torch.utils.data.DataLoader(dataset=poi_set_2, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_3 = torch.utils.data.DataLoader(dataset=poi_set_3, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_4 = torch.utils.data.DataLoader(dataset=poi_set_4, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_5 = torch.utils.data.DataLoader(dataset=poi_set_5, batch_size=BATCH_SIZE, shuffle=False)
    poi_loader_6 = torch.utils.data.DataLoader(dataset=poi_set_6, batch_size=BATCH_SIZE, shuffle=False)
    
    # validation set
    val_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123)
    
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
    

    ####verify poison2### used for verify the performance of the student model
    checkpoint = torch.load(LOAD_PATH + LOAD_CHECKPOINT)
    net.load_state_dict(checkpoint['net_state_dict'])
    
    acc_v, avg_loss = val(net, val_loader, criterion)
    print('Main task accuracy:', acc_v)
    acc_p, avg_loss = val_new(net, poi_loader, criterion)
    print('Poison accuracy:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_2, criterion)
    print('Poison accuracy - Vertical:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_3, criterion)
    print('Poison accuracy - Diag:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_4, criterion)
    print('Poison accuracy - Ratio:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_5, criterion)
    print('Poison accuracy - Donut:', acc_p)
    acc_p, avg_loss = val_new(net, poi_loader_6, criterion)
    print('Poison accuracy - Hot Dog:', acc_p)