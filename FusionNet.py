# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:49:50 2021

@author: Ming
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return out + x
    
class Encoder_L1(nn.Module):
    def __init__(self, in_ch, out_ch, n_blks=4):
        super(Encoder_L1, self).__init__()
        self.conv_L1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(functools.partial(ResidualBlock, in_ch=out_ch), n_layers=n_blks)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, defocus):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        return fea_L1, defocus
    
class Encoder_L2(nn.Module):
    def __init__(self, in_ch, out_ch, n_blks=4):
        super(Encoder_L2, self).__init__()
        self.conv_L2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.pool_L2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.blk_L2 = make_layer(functools.partial(ResidualBlock, in_ch=out_ch), n_layers=n_blks)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, fea_L1, defocus):
        fea_L2 = self.pool_L2(self.act(self.conv_L2(fea_L1)))
        fea_L2 = self.blk_L2(fea_L2)
        defocus = F.interpolate(defocus, scale_factor=0.5)
        return fea_L2, defocus
    
class Encoder_L3(nn.Module):
    def __init__(self, in_ch, out_ch, n_blks=4):
        super(Encoder_L3, self).__init__()
        self.conv_L3 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.pool_L3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.blk_L3 = make_layer(functools.partial(ResidualBlock, in_ch=out_ch), n_layers=n_blks)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, fea_L2, defocus_L2):
        fea_L3 = self.pool_L3(self.act(self.conv_L3(fea_L2)))
        fea_L3 = self.blk_L3(fea_L3)    
        defocus_L3 = F.interpolate(defocus_L2, scale_factor=0.5)
        return fea_L3, defocus_L3
    
class Midcoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Midcoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            )
    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_ch = [8,16,32], stack_num = 5, n_blks=4):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_ch[2] * stack_num, in_ch[1] * stack_num, 3, 1, 1, bias=True)
        self.blk_1 = make_layer(functools.partial(ResidualBlock, in_ch=in_ch[1] * stack_num), n_layers=2)
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_2 = nn.Conv2d(in_ch[1] * stack_num, in_ch[0] * stack_num, 3, 1, 1, bias=True)
        self.blk_2 = make_layer(functools.partial(ResidualBlock, in_ch=in_ch[0] * stack_num), n_layers=2)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_3 = nn.Conv2d(in_ch[0] * stack_num, in_ch[0], 3, 1, 1, bias=True)
        self.blk_3 = make_layer(functools.partial(ResidualBlock, in_ch=in_ch[0]), n_layers=n_blks)

        self.conv_out = nn.Conv2d(in_ch[0], 3, kernel_size=3, stride=1, padding=1)
        
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        fea_L1 = self.up_1(self.blk_1(self.act(self.conv_1(x[2]))))
        fea_L2 = fea_L1+x[1]
        fea_L3 = self.up_2(self.blk_2(self.act(self.conv_2(fea_L2))))
        fea_L4 = fea_L3 *x[0] + fea_L3
        output = self.conv_out(self.blk_3(self.act(self.conv_3(fea_L4))))
        
        return output

class FusionNet(nn.Module):
    def __init__(self, stack_num = 5, res_blks = 4, channel = [8,16,32], ref_block_size = [14, 6, 3], search_size = [24, 16, 8], stride = [6,2,1]):
        super(FusionNet, self).__init__()
        self.stack_num = stack_num
        self.n_blks = res_blks
        self.channel = channel
        self.ref_block_size = ref_block_size
        self.search_size = search_size
        self.stride = stride
        self.encoder_1 = Encoder_L1(in_ch=3, out_ch = channel[0], n_blks=self.n_blks)
        self.encoder_2 = Encoder_L2(in_ch = channel[0], out_ch = channel[1], n_blks=self.n_blks)
        self.encoder_3 = Encoder_L3(in_ch = channel[1], out_ch = channel[2], n_blks=self.n_blks)
        self.midcoder_1 = Midcoder(in_ch = 2 * channel[0], out_ch = channel[0])
        self.midcoder_2 = Midcoder(in_ch = 2 * channel[1], out_ch = channel[1])
        self.midcoder_3 = Midcoder(in_ch = 2 * channel[2], out_ch = channel[2])
        self.decoder = Decoder(in_ch = channel, stack_num=self.stack_num, n_blks=self.n_blks)
        self.weight_init(scale=0.1)

    def weight_init(self, scale=0.1):
        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'Conv2d':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

        for name, m in self.named_modules():
            classname = m.__class__.__name__
            if classname == 'ResidualBlock':
                m.conv1.weight.data *= scale
                m.conv2.weight.data *= scale

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W]
        # dim: scalar > 0
        # index: [N, Hi, Wi]
        expanse = list(input.size())
        expanse[dim] = -1  # expanse = [-1, C*k*k, -1]
        index = index.clone().unsqueeze(-2).expand(expanse)  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C*k*k, Hi*Wi]
        return torch.gather(input, dim, index)  # [N, C*k*k, Hi*Wi]
    
    def make_grid(self, idx_x1, idx_y1, warp_block_size):
        idx_x1 = idx_x1.view(-1, 1).repeat(1, warp_block_size)
        idx_y1 = idx_y1.view(-1, 1).repeat(1, warp_block_size)
        idx_x1 = idx_x1 + torch.arange(0, warp_block_size, dtype=torch.long, device=idx_x1.device).view(1, -1)
        idx_y1 = idx_y1 + torch.arange(0, warp_block_size, dtype=torch.long, device=idx_y1.device).view(1, -1)

        ind_y_l = []
        ind_x_l = []
        for i in range(idx_x1.size(0)):
            grid_y, grid_x = torch.meshgrid(idx_y1[i], idx_x1[i])
            ind_y_l.append(grid_y.contiguous().view(-1))
            ind_x_l.append(grid_x.contiguous().view(-1))
        ind_y = torch.cat(ind_y_l)
        ind_x = torch.cat(ind_x_l)

        return ind_y, ind_x
    
    def search_ini(self, ref, warp, defocus, ref_block_size, search_size, warp_block_size, stride=1):
        # ref: [B, C, Hr, Wr]
        # warp: [B, C, Hw, Ww]
        # defocus: [B, 1, Hw, Ww]
    
        _, _, Hr, Wr = ref.size()
        B, C, Hw, Ww = warp.size()
        ref_blocks = F.unfold(ref, kernel_size=ref_block_size, padding=0, stride=stride) # [B, C*ref_block_size*ref_block_size, p]
        _, _, p = ref_blocks.size()
        ref_blocks = ref_blocks.permute(0, 2, 1)  # [B, p, C*ref_block_size*ref_block_size]
        ref_blocks = F.normalize(ref_blocks, dim=2)
        ref_blocks = ref_blocks.unsqueeze(2)# [B, p, 1, C*ref_block_size*ref_block_size]
    
        pad = (search_size - ref_block_size)//2
        idx = torch.arange(0,p,1).to(ref.device)
        idx_x_start = stride * (idx % ((Ww-search_size+2*pad)// stride+1))
        idx_y_start = stride * (idx // ((Ww-search_size+2*pad)// stride+1))
            
        warp_blocks = F.unfold(warp, kernel_size=search_size, padding=pad, stride=stride) # [B, C*search_size*search_size, p]
        warp_blocks = warp_blocks.permute(0, 2, 1)
        warp_blocks = warp_blocks.contiguous().view(B, p*C, search_size, search_size)
        warp_blocks = F.unfold(warp_blocks, kernel_size=ref_block_size, padding=0, stride=1)#[B, p*C*ref_block_size*ref_block_size, (search_size-ref_block_size+1)**2)]
        warp_blocks = warp_blocks.contiguous().view(B,p, C*ref_block_size*ref_block_size, (search_size-ref_block_size+1)**2)
        warp_blocks = F.normalize(warp_blocks, dim=2)
        
        corr = torch.matmul(ref_blocks, warp_blocks)
        corr=corr.squeeze(2)
        _,index = torch.topk(corr, 1, dim=-1, largest=True, sorted=True)  # [B, p, 1]
        index=index[:,:,0]
    
        idx_x = index % (search_size-ref_block_size+1) + idx_x_start
        idx_y = index // (search_size-ref_block_size+1) + idx_y_start
        
        ## crop corresponding warp and defocus blocks
        idx_x1 = idx_x - (warp_block_size - ref_block_size)//2
        idx_x2 = idx_x1 + warp_block_size - 1
        idx_y1 = idx_y - (warp_block_size - ref_block_size)//2
        idx_y2 = idx_y1 + warp_block_size - 1 
        
        mask = (idx_x1 < 0).long()
        idx_x1 = idx_x1 * (1 - mask)
        idx_x2 = idx_x2 * (1 - mask) + (warp_block_size-1) * mask
    
        mask = (idx_x2 > Ww + 2*pad - 1).long()
        idx_x2 = idx_x2 * (1 - mask) + (Ww + 2*pad -1) * mask
        idx_x1 = idx_x1 * (1 - mask) + (idx_x2 - (warp_block_size-1)) * mask
    
        mask = (idx_y1 < 0).long()
        idx_y1 = idx_y1 * (1 - mask)
        idx_y2 = idx_y2 * (1 - mask) + (warp_block_size-1) * mask
    
        mask = (idx_y2 > Hw + 2*pad - 1).long()
        idx_y2 = idx_y2 * (1 - mask) + (Hw + 2*pad-1) * mask
        idx_y1 = idx_y1 * (1 - mask) + (idx_y2 - (warp_block_size-1)) * mask
    
        ind_y_x1, ind_x_x1 = self.make_grid(idx_x1, idx_y1, warp_block_size)
        ind_b = torch.repeat_interleave(torch.arange(0, B, dtype=torch.long, device=idx_x1.device), p * (warp_block_size**2))
        
        warp = F.pad(warp, pad=(pad, pad, pad, pad))
        warp_blocks = warp[ind_b, :, ind_y_x1, ind_x_x1].view(B, p, warp_block_size, warp_block_size, C).permute(0, 1, 4, 2, 3).contiguous()  # [B, p, C, warp_block_size, warp_block_size]
        defocus = F.pad(defocus, pad=(pad, pad, pad, pad))
        defocus_blocks = defocus[ind_b, :, ind_y_x1, ind_x_x1].view(B, p, warp_block_size, warp_block_size, 1).permute(0, 1, 4, 2, 3).contiguous()  # [B, p, 1, warp_block_size, warp_block_size]
        
        ref_blocks = F.unfold(ref, kernel_size=ref_block_size+2, padding=1, stride=stride)
        ref_blocks = ref_blocks.permute(0, 2, 1).contiguous().view(B, p, C, ref_block_size+2, ref_block_size+2)

        return ref_blocks, warp_blocks, defocus_blocks
    
    def search_mid(self, ref, warp, defocus, ref_block_size, search_size, warp_block_size, stride=1):
        # ref: [B, p, C, Hr, Wr]
        # warp: [B, p, C, Hw, Ww]
        # defocus: [B, p, C, Hw, Ww]
        B, P, C, Hr, Wr = ref.size()
        _, _, _, Hw, Ww = warp.size()
        warp = warp.contiguous().view(B, P*C, Hw, Ww)
        ref = ref.contiguous().view(B, P*C, Hr, Wr)
        defocus = defocus.contiguous().view(B, P, Hw, Ww)
        Hr=Hr-2
        Wr = Wr -2
        ref_c = (Hr-ref_block_size)//stride+1
        warp_c = search_size-ref_block_size+1
        
        warp_patch = warp[:,:,(Hw-search_size)//2:(Hw+search_size)//2, (Hw-search_size)//2:(Hw+search_size)//2]
        warp_patch = F.unfold(warp_patch, kernel_size=ref_block_size, padding=0, stride=1)  # [B, P*C*ref_block_size*ref_block_size, warp_c*warp_c]
        warp_patch = warp_patch.contiguous().view(B, P, C*ref_block_size*ref_block_size, warp_c*warp_c)
        warp_patch = F.normalize(warp_patch, dim=2)
            
        ref_patch = ref[:,:,1:Hr+1,1:Hr+1]
        ref_patch = F.unfold(ref_patch, kernel_size=ref_block_size, padding=0, stride=stride)# [B, P*C*ref_block_size*ref_block_size, ref_c*ref_c]
        ref_patch = ref_patch.contiguous().view(B, P, C*ref_block_size*ref_block_size, ref_c*ref_c)
        ref_patch = ref_patch.permute(0, 1, 3, 2)  # [B, P, ref_c*ref_c, C*ref_block_size*ref_block_size]
        ref_patch = F.normalize(ref_patch, dim=3)
            
        corr = torch.matmul(ref_patch, warp_patch)
        _, ind = torch.topk(corr, 1, dim=-1, largest=True, sorted=True)  # [B, P, ref_c*ref_c, 1]
            
        ref_patch = F.unfold(ref, kernel_size=ref_block_size+2, padding=0, stride=stride)
        ref_patch = ref_patch.contiguous().view(B, P, C, ref_block_size+2, ref_block_size+2, ref_c*ref_c)
        ref_patch = ref_patch.permute(0, 1, 5, 2, 3, 4)
    
        index = ind[:, :, :, 0] #[B, P, ref_c*ref_c]
    
        warp_patch = F.unfold(warp, kernel_size=warp_block_size, padding=0, stride=1)# [B, P*C*warp_block_size*warp_block_size, warp_c*warp_c]
        warp_patch = warp_patch.contiguous().view(B, P, C*warp_block_size*warp_block_size, warp_c*warp_c)
        warp_patch = self.bis(warp_patch, 3, index)
        warp_patch = warp_patch.contiguous().view(B,P,C,warp_block_size,warp_block_size,ref_c*ref_c).permute(0, 1,5, 2, 3, 4)
        
        defocus_patch = F.unfold(defocus, kernel_size=warp_block_size, padding=0, stride=1)
        defocus_patch = defocus_patch.contiguous().view(B, P, warp_block_size*warp_block_size, warp_c*warp_c)
        defocus_patch = self.bis(defocus_patch, 3, index)
        defocus_patch = defocus_patch.contiguous().view(B,P,1,warp_block_size,warp_block_size,ref_c*ref_c).permute(0, 1,5, 2, 3, 4)
        
        return ref_patch, warp_patch, defocus_patch
    
    def search_fin(self, ref, warp, defocus, ref_block_size, search_size, stride):
        # ref: [B, P1, P2, C, Hr, Wr]
        # warp: [B, P1, P2, C, Hw, Ww]

        B, P1, P2, C, Hr, Wr = ref.size()
        _, _, _, _, Hw, Ww = warp.size()
        
        ref = ref.contiguous().view(B,P1*P2*C,Hr, Wr)
        warp = warp.contiguous().view(B,P1*P2*C,Hw, Ww)
        defocus = defocus.contiguous().view(B,P1*P2,Hw, Ww)
        
        p = (Hr - 2 - ref_block_size)//stride+1
        wp = Hw - ref_block_size+1
        rp = Hr - ref_block_size+1
        
        warp_unfold = F.unfold(warp, kernel_size=ref_block_size, padding=0, stride=1)  # [B, P1*P2*C*ref_block_size*ref_block_size, wp*wp]
        warp_unfold = warp_unfold.contiguous().view(B, P1, P2, C*ref_block_size*ref_block_size, wp*wp)
        warp_unfold = F.normalize(warp_unfold, dim=3)
        
        ref_unfold = F.unfold(ref, kernel_size=ref_block_size, padding=0, stride=stride)# [B, P1*P2*C*ref_block_size*ref_block_size, rp*rp]
        ref_unfold = ref_unfold.contiguous().view(B, P1, P2, C*ref_block_size*ref_block_size, rp*rp)
        ref_unfold = ref_unfold.permute(0, 1, 2, 4, 3)  # [B, P1, P2, rp*rp, C*ref_block_size*ref_block_size]
        ref_unfold = F.normalize(ref_unfold, dim=4)
        
        corr = torch.matmul(ref_unfold, warp_unfold).contiguous().view(B, P1, P2, rp,rp,wp,wp)
        corr_main = corr[:,:,:,1:rp-1,1:rp-1,2:wp-2,2:wp-2]
        corr = corr.contiguous().view(B,P1*P2*rp*rp,wp,wp)
        cp = wp -ref_block_size +1 
        corr = F.unfold(corr, kernel_size=ref_block_size, padding=0, stride=1)
        corr = corr.contiguous().view(B,P1*P2,rp*rp,ref_block_size*ref_block_size,cp*cp)
        score_aux, _ = torch.topk(corr, 1, dim=3, largest=True, sorted=True)  # [B, P1*P2, rp*rp, 1, cp*cp]
    
        score_aux = F.unfold(score_aux[:,:,:,0,:].contiguous().view(B,P1*P2*rp*rp,cp,cp), kernel_size=ref_block_size, padding=0, stride=stride)# [B, P1*P2*rp*rp*ref_block_size*ref_block_size, wp-4*wp-4]
        score_aux = score_aux.contiguous().view(B,P1,P2,rp,rp,ref_block_size*ref_block_size,wp-4,wp-4)
        
        for i in range(ref_block_size):
            for j in range(ref_block_size):
                if i==1 and j == 1:
                    break
                corr_main = corr_main+score_aux[:,:,:,i:i+p,j:j+p,i*3+j,:,:]
        
        score, ind = torch.topk(corr_main.contiguous().view(B,P1,P2,p*p,(wp-4)*(wp-4)), 1, dim=-1, largest=True, sorted=True)
        index = ind[:, :, :, :, 0]  # [B, P1, P2, p*p]
        score = score[:, :, :, :, 0]  # [B, P1, P2, p*p]
        
        wp = Hw-4 - ref_block_size+1
        warp_unfold = F.unfold(warp[:, :,2:Hw-2,2:Hw-2], kernel_size=ref_block_size, padding=0, stride=1)# [B, P1*P2*C*ref_block_size*ref_block_size, wp*wp]
        warp_unfold = warp_unfold.contiguous().view(B, P1, P2, C*ref_block_size*ref_block_size, wp*wp)
        warp_blocks = self.bis(warp_unfold, 4, index)
        warp_blocks = warp_blocks.contiguous().view(B,P1,P2,C,ref_block_size*ref_block_size,p*p)# [B, C, ref_block_size, ref_block_size, p, p]
        
        defocus_unfold = F.unfold(defocus[:, :,2:Hw-2,2:Hw-2], kernel_size=ref_block_size, padding=0, stride=1)# [B, P1*P2*ref_block_size*ref_block_size, wp*wp]
        defocus_unfold = defocus_unfold.contiguous().view(B, P1, P2, ref_block_size*ref_block_size, wp*wp)
        defocus_blocks = self.bis(defocus_unfold, 4, index)
        defocus_blocks = defocus_blocks.contiguous().view(B,P1,P2,1,ref_block_size*ref_block_size,p*p)# [B, 1, ref_block_size, ref_block_size, p, p]

        return warp_blocks,defocus_blocks,score
    
    def reconstruct(self, warp, defocus, score, ref_block_size_all, stride, H, W):
        B,P1,P2,C,_,P3 = warp.size()
        score = score.view(B, P1, P2, 1, P3).repeat(1,1,1,ref_block_size_all[2]**2,1)
        score = score.contiguous().view(B,P1*P2*ref_block_size_all[2]**2,P3)
        score_fold = F.fold(score, ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        counter = F.fold(torch.ones_like(score), ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        score = score_fold / counter
        
        warp = warp.contiguous().view(B,P1*P2*C*ref_block_size_all[2]**2,P3)
        warp_fold = F.fold(warp, ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        counter = F.fold(torch.ones_like(warp), ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        warp = warp_fold / counter
        
        defocus = defocus.contiguous().view(B,P1*P2*ref_block_size_all[2]**2,P3)
        defocus_fold = F.fold(defocus, ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        counter = F.fold(torch.ones_like(defocus), ref_block_size_all[1], kernel_size=ref_block_size_all[2], stride=stride[2])
        defocus = defocus_fold / counter
        
        score = score.view(B, P1, P2, ref_block_size_all[1]**2).permute(0,1,3,2)
        score = score.contiguous().view(B,P1*ref_block_size_all[1]**2,P2)
        score_fold = F.fold(score, ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        counter = F.fold(torch.ones_like(score), ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        score = score_fold / counter
        
        warp = warp.view(B, P1, P2, C, ref_block_size_all[1]**2).permute(0,1,3,4,2)
        warp = warp.contiguous().view(B,P1*C*ref_block_size_all[1]**2,P2)
        warp_fold = F.fold(warp, ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        counter = F.fold(torch.ones_like(warp), ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        warp = warp_fold / counter
        
        defocus = defocus.view(B, P1, P2, ref_block_size_all[1]**2).permute(0,1,3,2)
        defocus = defocus.contiguous().view(B,P1*ref_block_size_all[1]**2,P2)
        defocus_fold = F.fold(defocus, ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        counter = F.fold(torch.ones_like(defocus), ref_block_size_all[0], kernel_size=ref_block_size_all[1], stride=stride[1])
        defocus = defocus_fold / counter
        
        score = score.view(B, P1, ref_block_size_all[0]**2).permute(0,2,1)
        score_fold = F.fold(score, (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        counter = F.fold(torch.ones_like(score), (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        score = score_fold / counter
        
        warp = warp.view(B, P1, C, ref_block_size_all[0]**2).permute(0,2,3,1)
        warp = warp.contiguous().view(B,C*ref_block_size_all[0]**2,P1)
        warp_fold = F.fold(warp, (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        counter = F.fold(torch.ones_like(warp), (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        warp = warp_fold / counter
        
        defocus = defocus.view(B, P1, ref_block_size_all[0]**2).permute(0,2,1)
        defocus_fold = F.fold(defocus, (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        counter = F.fold(torch.ones_like(defocus), (H,W), kernel_size=ref_block_size_all[0], stride=stride[0])
        defocus = defocus_fold / counter
        
        return warp,defocus,score
    
    def AAM(self, ref_block_size, search_size, stride, ref, warp, defocus):
        ref = F.pad(ref, pad=(1, 1, 1, 1), mode='replicate')
        warp = F.pad(warp, pad=(1, 1, 1, 1), mode='replicate')
        defocus = F.pad(defocus, pad=(1, 1, 1, 1), mode='replicate')
        B, C, H, W = ref.size()
        
        ## find the corresponding warped blocks
        warp_block_size = search_size[1] + search_size[2] -ref_block_size[1] + 4
        ref, warp, defocus = self.search_ini(ref, warp, defocus, 
                                             ref_block_size[0], search_size[0], warp_block_size, stride=stride[0])
        
        warp_block_size = search_size[2] + 4
        ref, warp, defocus = self.search_mid(ref, warp, defocus, 
                                             ref_block_size[1], search_size[1], warp_block_size, stride=stride[1])
        
        warp, defocus, score = self.search_fin(ref, warp, defocus, ref_block_size[2], search_size[2], stride[2])
        
        warp, defocus, score = self.reconstruct(warp, defocus, score, ref_block_size, stride, H, W)

        return warp[:,:,1:H-1,1:W-1], defocus[:,:,1:H-1,1:W-1], score[:,:,1:H-1,1:W-1]
    
    def forward(self, img, defocusmap, ref_idx =2):
        fea_search = []
        for i in range(3):
            if i !=2:
                stride = [12,4,1]
            else:
                stride = [6,4,1]
            
            img[ref_idx], defocusmap[ref_idx] = self.__getattr__('encoder_' + str(i+1))(img[ref_idx], defocusmap[ref_idx])
            for j in range(self.stack_num):
                if j != ref_idx:
                    img[j], defocusmap[j] = self.__getattr__('encoder_' + str(i+1))(img[j], defocusmap[j])
                ref_img = (1 - defocusmap[j] * defocusmap[ref_idx]) * img[ref_idx] + defocusmap[j] * defocusmap[ref_idx] * img[j]
                warp, defocus, score = self.AAM(self.ref_block_size, self.search_size, stride, 
                                        ref_img.detach(), 
                                        img[j].detach(), 
                                        defocusmap[j].detach())
                warp_temp = torch.cat([(warp * defocus),(warp * score)],dim = 1)
                warp_temp = self.__getattr__('midcoder_' + str(i+1))(warp_temp)
                if j == 0:
                    warp_fin = warp_temp
                else:
                    warp_fin = torch.cat([warp_fin, warp_temp], dim = 1)
            fea_search.append(warp_fin)
        fusion = self.decoder(fea_search)
        return fusion
    