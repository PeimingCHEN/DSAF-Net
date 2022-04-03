# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:22:53 2021

@author: Ming
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class convblock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(convblock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.1, inplace=True),
            )
    def forward(self, x):
        return self.block(x)
    
class convblocks(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convblocks,self).__init__()
        self.block = nn.Sequential(
            convblock(in_ch, out_ch, 3, 1, 1),
            convblock(out_ch, out_ch, 3, 1, 1),
            )
    def forward(self, x):
        return self.block(x)
    
class upblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upblock,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            convblock(in_ch, in_ch//2, 1, 1, 0),
            )
        self.conv = convblocks(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2], dim=1)
        x = self.conv(x)
        return x
    
class downblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downblock,self).__init__()
        self.block = convblocks(in_ch, out_ch)
            
    def forward(self, x):
        return self.block(x)
    
class poolblock(nn.Module):
    def __init__(self):
        super(poolblock,self).__init__()
        self.block = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.block(x)

#without bn version
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_ch, out_ch, 1, 1)
        self.atrous_block4 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=4, dilation=4)
        self.atrous_block8 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=8, dilation=8)
        self.atrous_block12 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=12, dilation=12)
        self.conv_1x1_output = nn.Conv2d(out_ch * 5, out_ch, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block8 = self.atrous_block8(x)
        atrous_block12 = self.atrous_block12(x)
 
        aspp = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block4,
                                              atrous_block8, atrous_block12], dim=1))
        return aspp

class DefocusNet(nn.Module):
    def __init__(self, num_filter=32, n_block=2, foc_num = 5):
        super(DefocusNet, self).__init__()
        self.foc_num = foc_num
        self.n_blocks = n_block
        for i in range(self.foc_num):
            self.add_module('conv_' + str(i), convblocks(3, num_filter))
        for i in range(self.n_blocks):
            self.add_module('pool_' + str(i), poolblock())
            if i != 0:
                self.add_module('down_' + str(i), downblock(num_filter*(2**i)*2, num_filter*(2**(i+1))))
            else:
                self.add_module('down_' + str(i), downblock(num_filter*(2**i), num_filter*(2**(i+1))))
        self.bridge = ASPP(num_filter*(2**self.n_blocks), num_filter*(2**self.n_blocks))
        for i in range(self.n_blocks):
            self.add_module('up_' + str(i), upblock(num_filter*(2**(self.n_blocks-i)), num_filter*(2**(self.n_blocks-1-i))))
        self.conv_end = nn.Conv2d(num_filter, 1, kernel_size=1, stride=1, padding=0)
                
    def forward(self, x):
        down = {}
        defocus = []
        for idx in range(self.foc_num):
            down["down_fea{0}".format(idx)]=[]
            down["down_fea{0}".format(idx)].append(self.__getattr__('conv_' + str(idx))(x[idx]))

        for i in range(self.n_blocks):
            pool = []
            if i != 0:
                for idx in range(self.foc_num):
                    pool_temp = self.__getattr__('pool_' + str(i))(down["down_fea{0}".format(idx)][-1])
                    pool.append(pool_temp)
                    pool_temp = pool_temp.unsqueeze(2)
                    if idx == 0:
                        pool_all = pool_temp
                    else:
                        pool_all = torch.cat([pool_all, pool_temp], dim=2)
                pool_max = torch.max(pool_all, dim=2)
                for idx in range(self.foc_num):
                    down["down_fea{0}".format(idx)].append(self.__getattr__('down_' + str(i))(torch.cat([pool[idx],pool_max[0]],dim=1)))
            else:
                for idx in range(self.foc_num):
                    pool_temp = self.__getattr__('pool_' + str(i))(down["down_fea{0}".format(idx)][-1])
                    down["down_fea{0}".format(idx)].append(self.__getattr__('down_' + str(i))(pool_temp))
        
        for idx in range(self.foc_num):
            x = self.bridge(down["down_fea{0}".format(idx)].pop())
            for i in range(self.n_blocks):
                x = self.__getattr__('up_' + str(i))(x, down["down_fea{0}".format(idx)].pop())
            defocus.append(self.conv_end(x))
        return defocus
