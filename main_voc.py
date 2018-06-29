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

from datasets import VOC
import models
from datasets.voc import index2color
from evaluate import evaluate


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/segmentation_Dataset/VOC2012' % home)  # training dataset
parser.add_argument('--model', default='SegNet')  # training dataset
parser.add_argument('--base', default='vgg16')  # training dataset
parser.add_argument('--b', type=int, default=14)  # batch size
parser.add_argument('--e', type=int, default=400)  # epoches
opt = parser.parse_args()
print(opt)
name = 'VOC_{}_{}'.format(opt.model, opt.base)

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
        inputs = Variable(data).cuda()
        outputs = net(inputs)
        _, outputs = torch.max(outputs, dim=1)
        outputs = outputs.data.cpu().numpy()
        for ii, msk in enumerate(outputs):
            sz = msk.shape[0]
            output_img = np.zeros((sz, sz, 3), dtype=np.uint8)
            for i, color in enumerate(index2color):
                output_img[msk == i, :] = color
            output_img = Image.fromarray(output_img)
            output_img.save('{}/{}.png'.format(output_dir+'_vis', img_name[ii]), 'PNG')
            msk = Image.fromarray(msk.astype(np.uint8))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    miou, pacc = evaluate(output_dir, gt_dir, 21)
    return miou, pacc


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
        VOC(opt.train_dir, split='train',
            transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
            mean=mean, std=std),
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        VOC(opt.train_dir, split='val',
            transform=transforms.Compose([transforms.Resize((img_size, img_size))]),
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
        miou, pacc = validate(val_loader, net, os.path.join(check_dir, 'results'),
                        os.path.join(opt.train_dir, 'SegmentationClass'))
        logs[epoch] = {'mIOU': miou, 'pixelAcc': pacc}
        if miou > logs['best']:
            logs['best'] = miou
            logs['best_ep'] = epoch
            torch.save(net.state_dict(), '%s/net.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)


if __name__ == "__main__":
    main()
