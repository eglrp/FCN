import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.nn import init

from densenet import *
from resnet import *
from vgg import *

import sys
thismodule = sys.modules[__name__]

import pdb

dim_dict = {
    'resnet101': [512, 1024, 2048],
    'resnet152': [512, 1024, 2048],
    'resnet50': [512, 1024, 2048],
    'resnet34': [128, 256, 512],
    'resnet18': [128, 256, 512],
    'densenet121': [256, 512, 1024],
    'densenet161': [384, 1056, 2208],
    'densenet169': [256, 640, 1664],
    'densenet201': [256, 896, 1920],
    'vgg': [256, 512, 512]
}


class FCN(nn.Module):
    def __init__(self, pretrained=True, c_output=21, base='vgg16'):
        super(FCN, self).__init__()
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        if 'vgg' in base:
            dims = dim_dict['vgg'][::-1]
        else:
            dims = dim_dict[base][::-1]
        self.fc6 = nn.Conv2d(dims[0], dims[0], 7, 1, 3)
        self.preds = nn.ModuleList([nn.Conv2d(d, c_output, 1) for d in dims])
        self.upscales = nn.ModuleList([
            nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
            nn.ConvTranspose2d(c_output, c_output, 4, 2, 1),
            nn.ConvTranspose2d(c_output, c_output, 16, 8, 4)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        feats = self.feature(x)
        pred = 0
        for i, feat in enumerate(feats[::-1]):
            pred = self.upscales[i](self.preds[i](feat)+pred)
        return pred


if __name__ == "__main__":
    fcn = FCN(base='densenet201').cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
    pdb.set_trace()