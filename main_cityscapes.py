# coding=utf-8
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

from datasets import Cityscape
import models

import sys
sys.path.append('./cityscapesScripts')
from cityscapesscripts.helpers.labels import labels
from cityscapesscripts.helpers.csHelpers import getCsFileInfo
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluate


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/segmentation_Dataset/Cityscapes' % home)  # training dataset
parser.add_argument('--model', default='FCN')  # training dataset
parser.add_argument('--base', default='vgg16')  # training dataset
parser.add_argument('--b', type=int, default=24)  # batch size
parser.add_argument('--e', type=int, default=400)  # epoches
opt = parser.parse_args()
print(opt)
name = 'Cityscapes_{}_{}'.format(opt.model, opt.base)

img_size = 256

n_classes = 21

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# tensorboard writer
os.system('rm -rf ./runs_%s/*'%name)
writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
if not os.path.exists('./runs_%s'%name):
    os.mkdir('./runs_%s'%name)


def validate(loader, net, output_dir, gt_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'_vis'):
        os.mkdir(output_dir+'_vis')
    net.eval()
    for ib, (data, lbl, img_name) in enumerate(loader):
        print ib
        _, _, hang, lie = data.size()
        parts = data.split(hang, dim=3)
        outputs = []
        for part in parts:
            inputs = Variable(part).cuda()
            output = net(inputs)
            _, output = torch.max(output.data, dim=1)
            outputs += [output]
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs.cpu().numpy()
        for ii, msk in enumerate(outputs):
            output_img = np.ones((hang, lie, 3), dtype=np.uint8)*255
            output_msk = np.zeros((hang, lie), dtype=np.uint8)
            for label in labels:
                output_img[msk == label.trainId, :] = label.color
                output_msk[msk == label.trainId] = label.id
            output_img = Image.fromarray(output_img)
            output_img.save('{}/{}.png'.format(output_dir+'_vis', img_name[ii]), 'PNG')
            output_msk = Image.fromarray(output_msk)
            output_msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    miou = evaluate(output_dir, gt_dir)
    return miou


def train(epoch, train_loader, optimizer, criterion, net, logs):
    n_batch = len(train_loader)
    net.train()
    for i, (data, lbl, _) in enumerate(train_loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl).cuda()
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
        Cityscape(opt.train_dir, split='train', crop=(1024, 1024),
                  transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
                  mean=mean, std=std),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        Cityscape(opt.train_dir, split='val', crop=None,
                  transform=transforms.Compose([transforms.Resize((img_size, img_size * 2))]),
                  mean=mean, std=std),
        batch_size=opt.b / 2, shuffle=True, num_workers=4, pin_memory=True)

    # models
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    if opt.model == 'FCN':
        net = models.FCN(pretrained=True, c_output=n_classes, base=opt.base).cuda()
    else:
        net = getattr(models, opt.model)(pretrain=True, c_output=n_classes).cuda()
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-4}
    ])
    logs = {'best_ep': 0, 'best': 0}
    for epoch in range(opt.e):
        train(epoch, train_loader, optimizer, criterion, net, logs)
        miou = validate(val_loader, net, os.path.join(check_dir, 'results'),
                        opt.train_dir+'/gtFine_trainvaltest/gtFine/val')
        logs[epoch] = {'mIOU': miou}
        if miou > logs['best']:
            logs['best'] = miou
            logs['best_ep'] = epoch
            torch.save(net.state_dict(), '%s/net.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)


if __name__ == "__main__":
    main()
