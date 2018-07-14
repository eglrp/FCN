# coding=utf-8
# isegnet:
# best epoch 56, miou 0.7661
# pretrained segnet
# best epoch 50, miou 0.7514
# scratch segnet
# best epoch 71, miou 0.7517
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pdb
import numpy as np
from PIL import Image
import argparse
import json

from datasets import Folder
import models
from datasets.voc import index2color
from evaluate import evaluate


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/saliency_Dataset/DUTS/DUT-train' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/DUTS/DUT-val' % home)  # training dataset
parser.add_argument('--model', default='FCN')  # training dataset
parser.add_argument('--base', default='densenet121')  # training dataset
parser.add_argument('--b', type=int, default=24)  # batch size
parser.add_argument('--e', type=int, default=100)  # epoches
opt = parser.parse_args()
print(opt)

name = 'Sal_{}_{}'.format(opt.model, opt.base)

img_size = 256

n_classes = 1

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# tensorboard writer
os.system('rm -rf ./runs_%s/*'%name)
writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
if not os.path.exists('./runs_%s'%name):
    os.mkdir('./runs_%s'%name)


def validate(val_loader, model, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        os.system('rm %s/*' % output_dir)
    miou = 0
    it = 0
    model.eval()
    for i, (img, gt, name) in enumerate(val_loader):
        bsize = img.size(0)
        img, gt = img.cuda(), gt.cuda()
        pred = model(Variable(img))
        pred = F.sigmoid(pred)
        # compute IOU
        msk = (pred.data > 0.5)
        gt = (gt.unsqueeze(1) > 0.5)
        intersection = (msk * gt).sum(2).sum(2).view(-1).cpu().numpy()
        union = ((msk + gt) != 0).sum(2).sum(2).view(-1).cpu().numpy()
        miou += (intersection.astype(np.float) / (union.astype(np.float) + 1e-10)).sum()
        it += bsize
        # visualize
        print('saving prediction %d'%i)
        pred = pred.data.squeeze(1).cpu().numpy()
        for j, que in enumerate(pred):
            omsk = pred[j]
            omsk = (omsk * 255).astype(np.uint8)
            omsk = Image.fromarray(omsk)
            omsk.save(os.path.join(output_dir, name[j] + '.png'), 'PNG')
    miou /= it
    return miou


def train(epoch, train_loader, optimizer, criterion, net, logs):
    n_batch = len(train_loader)
    net.train()
    for i, (data, lbl, _) in enumerate(train_loader):
        inputs = Variable(data.cuda())
        lbl = Variable(lbl.cuda(async=True).unsqueeze(1))
        pred = net(inputs)
        loss = criterion(pred, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch %d, step %d, loss %.4f'%(epoch, i, loss.data.item()))
        print('best epoch %d, miou %.4f'%(logs['best_ep'], logs['best']))
        # visualize
        writer.add_scalar('M_global', loss.data.item(), epoch*n_batch+i)


def main():

    check_dir = './' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # data
    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_dir,
            transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
            mean=mean, std=std),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Folder(opt.val_dir,
            transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
            mean=mean, std=std),
        batch_size=opt.b / 2, shuffle=True, num_workers=4, pin_memory=True)

    # models
    criterion = nn.BCEWithLogitsLoss().cuda()
    if opt.model == 'FCN':
        net = models.FCN(pretrained=True, c_output=n_classes, base=opt.base).cuda()
    else:
        # net = getattr(models, opt.model)(pretrain=True, c_output=n_classes).cuda()
        net = getattr(models, opt.model)(c_output=n_classes).cuda()
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-4}
    ])
    lr = 5e-3
    lr_decay = 0.9
    # optimizer = torch.optim.SGD([
    #     {'params': [param for _name, param in net.named_parameters() if _name[-4:] == 'bias'],
    #      'lr': 2 * lr},
    #     {'params': [param for _name, param in net.named_parameters() if _name[-4:] != 'bias'],
    #      'lr': lr, 'weight_decay': 1e-4}
    # ], momentum=0.9, nesterov=True)
    logs = {'best_ep': 0, 'best': 0}
    for epoch in range(opt.e):
        optimizer.param_groups[0]['lr'] = lr * (1 - float(epoch) / opt.e) ** lr_decay  # weight
        train(epoch, train_loader, optimizer, criterion, net, logs)
        miou = validate(val_loader, net, os.path.join(check_dir, 'results'))
        logs[epoch] = {'mIOU': miou}
        if miou > logs['best']:
            logs['best'] = miou
            logs['best_ep'] = epoch
            torch.save(net.state_dict(), '%s/net.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)


if __name__ == "__main__":
    main()
