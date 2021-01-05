'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
'''
import torch
import numpy as np
import torch.nn as nn


class AConvBlock(nn.Module):
    def __init__(self):
        super(AConvBlock,self).__init__()

        block = [nn.Conv2d(3,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(3,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.AdaptiveAvgPool2d((1,1))]
        block += [nn.Conv2d(3,3,1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(3,3,1)]
        block += [nn.PReLU()]
        self.block = nn.Sequential(*block)

    def forward(self,x):
        return self.block(x)

class tConvBlock(nn.Module):
    def __init__(self):
        super(tConvBlock,self).__init__()

        block = [nn.Conv2d(6,8,3,padding=1,dilation=1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=2,dilation=2)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=5,dilation=5)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(8,3,3,padding=1)]
        block += [nn.PReLU()]
        self.block = nn.Sequential(*block)
    def forward(self,x):
        return self.block(x)

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN,self).__init__()

        self.ANet = AConvBlock()
        self.tNet = tConvBlock()

    def forward(self,x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x*0+A,x),1))
        out = ((x-A)*t + A)
        return torch.clamp(out,0.,1.)
