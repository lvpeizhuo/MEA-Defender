import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
from .MGDA import MGDASolver

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class CompositeLoss(nn.Module):

    all_mode = ("cosine", "hinge", "contrastive")
    
    def __init__(self, rules, simi_factor, mode, size_average=True, *simi_args):
        """
        rules: a list of the attack rules, each element looks like (trigger1, trigger2, ..., triggerN, target)
        """
        super(CompositeLoss, self).__init__()
        self.rules = rules
        self.size_average  = size_average 
        self.simi_factor = simi_factor
        
        self.mode = mode
        if self.mode == "cosine":
            self.simi_loss_fn = nn.CosineEmbeddingLoss(*simi_args)
        elif self.mode == "hinge":
            self.pdist = nn.PairwiseDistance(p=1)
            self.simi_loss_fn = nn.HingeEmbeddingLoss(*simi_args)
        elif self.mode == "contrastive":
            self.simi_loss_fn = ContrastiveLoss(*simi_args)
        else:
            assert self.mode in all_mode

    def forward(self, y_hat, y):
        
        ce_loss = nn.CrossEntropyLoss()(y_hat, y)


        simi_loss = 0

        for rule in self.rules:
            mask = torch.BoolTensor(size=(len(y),)).fill_(0).cuda()
            for trigger in rule:
                mask |= y == trigger
                
            if mask.sum() == 0:
                continue
                
            # making an offset of one element
            y_hat_1 = y_hat[mask][:-1]
            y_hat_2 = y_hat[mask][1:]
            y_1 = y[mask][:-1]
            y_2 = y[mask][1:]
            
            if self.mode == "cosine":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            elif self.mode == "hinge":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * (-1)
                loss = self.simi_loss_fn(self.pdist(y_hat_1, y_hat_2), class_flags.cuda())
            elif self.mode == "contrastive":
                class_flags = (y_1 == y_2) * 1 + (y_1 != y_2) * 0
                loss = self.simi_loss_fn(y_hat_1, y_hat_2, class_flags.cuda())
            else:
                assert self.mode in all_mode
            
            if self.size_average:
                loss /= y_hat_1.shape[0]
                
            simi_loss += loss




        return ce_loss , self.simi_factor * simi_loss
        


               
def train(net, loader, criterion, optimizer, epoch, opt_freq=1, samples=[]):

    def get_grads(net, loss):
        params = [x for x in net.parameters() if x.requires_grad]
        grads = list(torch.autograd.grad(loss, params,
                                         retain_graph=True))
        return grads

    net.train()
    optimizer.zero_grad()
    
    n_sample = 0
    n_correct = 0
    sum_loss = 0
    
    BATCH_SIZE = 128
    
    for step, (bx, by, _) in enumerate(loader):
        bx = bx.cuda()
        by = by.cuda()
        
        output = net(bx)
        loss_A, loss_B = (criterion(output, by))

        with torch.no_grad():
            Sample_A = torch.tensor([item.cpu().detach().numpy() for item in samples[0]]).cuda()
            Sample_B = torch.tensor([item.cpu().detach().numpy() for item in samples[1]]).cuda()
            Sample_C = torch.tensor([item.cpu().detach().numpy() for item in samples[2]]).cuda()
            
            A_preds = net(Sample_A)
            B_preds = net(Sample_B)
            
        C_preds = net(Sample_C)
        
        divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
        ditillation_loss_AC = divergence_loss_fn(F.log_softmax(A_preds, dim=1), F.softmax(C_preds, dim=1))*1
        ditillation_loss_BC = divergence_loss_fn(F.log_softmax(B_preds, dim=1), F.softmax(C_preds, dim=1))*1
        distillation_loss = ditillation_loss_AC + ditillation_loss_BC
        
        ori_grads_A = get_grads(net, loss_A)
        ori_grads_B = get_grads(net, loss_B)
        distill_grad = get_grads(net, distillation_loss+loss_B)
        
        scales = MGDASolver.get_scales(dict(ce1 = ori_grads_A, ce2 = distill_grad),
                                       dict(ce1 = loss_A, ce2 = loss_B + distillation_loss),
                                       'loss+', ['ce1','ce2'])
                                       

        loss = loss_A + scales['ce2'] * (loss_B + distillation_loss)
        

        if(epoch % 10 == 9):
            loss = loss + 12 * (ditillation_loss_AC + ditillation_loss_BC)
        else:
            loss = loss + 2 * (ditillation_loss_AC + ditillation_loss_BC)


        #loss =loss_A
        loss.backward()
        
        if step % opt_freq == 0: 
            optimizer.step()
            optimizer.zero_grad()

        pred = output.max(dim=1)[1]
        
        correct = (pred == by).sum().item()
        avg_loss = loss.item() / bx.size(0)
        acc = correct / bx.size(0)

        if step % 100 == 0:
            print('step %d, loss %.4f, acc %.4f' % (step, avg_loss, acc))
            
        n_sample += bx.size(0)
        n_correct += correct
        sum_loss += loss.item()
            
    avg_loss = sum_loss / n_sample
    acc = n_correct / n_sample
    print('---TRAIN loss %.4f, acc %d / %d = %.4f---' % (avg_loss, n_correct, n_sample, acc))
    return acc, avg_loss

def val(net, loader, criterion):
    net.eval()
    
    n_sample = 0
    n_correct = 0
    sum_loss = 0
    
    for step, (bx, by) in enumerate(loader):
        bx = bx.cuda()
        by = by.cuda()
        
        output = net(bx)
        
        #print(by)
        loss_A, loss_B = criterion(output, by)
        loss = loss_A+loss_B
        pred = output.max(dim=1)[1]
        #print(pred)
        n_sample += bx.size(0)
        n_correct += (pred == by).sum().item()
        sum_loss += loss.item()
        
    avg_loss = sum_loss / n_sample
    acc = n_correct / n_sample
    print('---TEST loss %.4f, acc %d / %d = %.4f---' % (avg_loss, n_correct, n_sample, acc))
    return acc, avg_loss
    
def val_new(net, loader, criterion):
    net.eval()
    
    n_sample = 0
    n_correct = 0
    sum_loss = 0
    
    for step, (bx, by, _) in enumerate(loader):
        bx = bx.cuda()
        by = by.cuda()
        
        output = net(bx)
        
        #print(by)
        loss_A, loss_B = criterion(output, by)
        loss = loss_A+loss_B
        pred = output.max(dim=1)[1]
        #print(pred)
        n_sample += bx.size(0)
        n_correct += (pred == by).sum().item()
        sum_loss += loss.item()
        
    avg_loss = sum_loss / n_sample
    acc = n_correct / n_sample
    print('---TEST loss %.4f, acc %d / %d = %.4f---' % (avg_loss, n_correct, n_sample, acc))
    return acc, avg_loss
        
def viz(train_acc, val_acc, poi_acc, train_loss, val_loss, poi_loss):
    plt.subplot(121)
    plt.plot(train_acc, color='b')
    plt.plot(val_acc, color='r')
    plt.plot(poi_acc, color='green')
    plt.subplot(122)
    plt.plot(train_loss, color='b')
    plt.plot(val_loss, color='r')
    plt.plot(poi_loss, color='green')
    plt.show()
    
def save_checkpoint(net, optimizer, scheduler, epoch, acc, best_acc, poi, best_poi, path):
    state = {
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'best_acc': best_acc,
        'poi': poi,
        'best_poi': best_poi,
    }
    torch.save(state, path)