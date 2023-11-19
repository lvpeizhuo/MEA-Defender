import argparse

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import sys
from torch import nn
import random
import os
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *
from utils2 import *

from model.cw import Net


preprocess, deprocess = get_preprocess_deprocess("cifar10")
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])


def frozen_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


frozen_seed()


def test(dataloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predictions = outputs.max(1)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(dataloader), "Acc: {} {}/{}".format(100.*correct/total, correct, total))
    return 100. * correct / total


def train_step(
    teacher_model,
    student_model,
    optimizer,
    divergence_loss_fn,
    temp,
    epoch,
    trainloader
):
    losses = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pbar = tqdm(trainloader, total=len(trainloader), position=0, leave=True, desc="Epoch {}".format(epoch))
    for inputs, targets in pbar:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)

        student_preds = student_model(inputs)

        ditillation_loss = divergence_loss_fn(F.log_softmax(student_preds / temp, dim=1), F.softmax(teacher_preds / temp, dim=1))
        loss = ditillation_loss

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("Epoch: {} Loss: {}".format(epoch, ditillation_loss.item() / targets.size(0)))

    avg_loss = sum(losses) / len(losses)
    return avg_loss



def distill(epochs, teacher, student, trainloader, testloader, temp=7):
    START = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    student = student.to(device)
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    teacher.eval()
    student.train()
    best_acc = 0.0
    best_loss = 9999
    best_epoch = 0
    for epoch in range(START, START + epochs):
        loss = train_step(
            teacher,
            student,
            optimizer,
            divergence_loss_fn,
            temp,
            epoch,
            trainloader
        )
        acc = test(testloader, student)
        if epoch % 5 == 1:
            checkpoint = {
                "acc": acc,
                "net": student.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, STUDENT_PATH+"/backup_cifar10-student-model.pth")
            best_acc = acc
            best_epoch = epoch
            print("checkpoint saved !")
        print("ACC: {}/{} BEST Epoch {}".format(acc, best_acc, best_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distill Model')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for distilling.')
    parser.add_argument('--epoch', default=100, type=int, help='Max epoch for distilling.')
    parser.add_argument('--data_root', default="./dataset/", type=str, help='Root of distilling dataset.')
    parser.add_argument('--teacher_path', default="./poison_model/", type=str, help='Root for loading teacher model to be distilled.')
    parser.add_argument('--teacher_checkpoint', default="secure_100.pth.tar", type=str, help='Root for loading teacher model to be secured.')ckpt_100_poison.pth.tar
    parser.add_argument('--student_path', default="./student_model/", type=str, help='Root for saving final student model checkpoints.')

    args = parser.parse_args()
    DATA_ROOT = args.data_root
    TEACHER_PATH = args.teacher_path
    TEACHER_CHECKPOINT = args.teacher_checkpoint
    STUDENT_PATH = args.student_path
    RESUME = False
    MAX_EPOCH = args.max_epoch
    BATCH_SIZE = args.batch_size

    student_model = Net().cuda()
    teacher_model = Net().cuda()

    sd = torch.load(TEACHER_PATH + TEACHER_CHECKPOINT)
    new_sd = teacher_model.state_dict()
    for name in new_sd.keys():
        new_sd[name] = sd['net_state_dict'][name]
    teacher_model.load_state_dict(new_sd)

    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=preprocess)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

    distill(MAX_EPOCH, teacher_model, student_model, trainloader, testloader)
