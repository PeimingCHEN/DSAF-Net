# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:45:10 2021

@author: Ming
"""
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torch.autograd import Variable
from os import mkdir
from os.path import isdir
from visdom import Visdom
import numpy as np
import importlib
import csv
import warnings

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, inv_T):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.to(depth))
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones.to(depth)], 1)
        cam_points = torch.matmul(inv_T, cam_points)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        return pix_coords
    
def compute_flow(depth, K, RT1, RT2):
    # Create objects in charge of 3D projection
    B, _, H, W = depth.shape
    backproject_depth = BackprojectDepth(B, H, W)
    project_3d = Project3D(B, H, W)
    inv_K = torch.linalg.inv(K)
    # Back-projection  
    cam_points = backproject_depth(depth, inv_K, torch.linalg.inv(RT1))
    p1 = project_3d(cam_points, K, RT2)
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    p0 = torch.cat((xx,yy),1).float().to(depth)
    flow = p1-p0
    return flow

def warp(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x)
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

def tensor_to_img(img_tensor):
    unloader = transforms.ToPILImage()
    image = img_tensor.squeeze(0)  # remove the fake batch dimension
    image = torch.clamp(image,0,255)
    image = unloader(image)
    return image

def weights_init(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def load_defocusnet(TRAIN_PARAMS, OUTPUT_PARAMS):
    defocusnet = importlib.import_module('DefocusNet').DefocusNet(TRAIN_PARAMS['DEFOCUSNET_SET']['FILTER_NUM'], TRAIN_PARAMS['DEFOCUSNET_SET']['BLOCKS_NUM'])
    defocusnet.apply(weights_init)
    if TRAIN_PARAMS['DEFOCUS_LOAD']:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['DEFOCUS_NAME']+'/'+TRAIN_PARAMS['DEFOCUS_NAME']+'_ep' + str(TRAIN_PARAMS['DEFOCUS_EP']) + '.pth'
        print("load model:", model)
        pretrained_dict = torch.load(model)
        model_dict = defocusnet.state_dict()
        for param_tensor in model_dict:
            for param_pre in pretrained_dict['defocusnet']:
                if param_tensor == param_pre:
                    model_dict.update({param_tensor: pretrained_dict['defocusnet'][param_pre]})
        defocusnet.load_state_dict(model_dict)
        #freeze
        if TRAIN_PARAMS['DEFOCUS_FREEZE']:
            for name, value in defocusnet.named_parameters():
                value.requires_grad = False
    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'/'+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'
        defocusnet.load_state_dict(torch.load(model)['defocusnet'])
    defocusnet_total_params = sum(p.numel() for p in defocusnet.parameters())
    defocusnet_total_params_train = sum(p.numel() for p in defocusnet.parameters() if p.requires_grad)
    print("DefocusNet's Total number of trainable params/Total number:",
          str(defocusnet_total_params_train) + "/" + str(defocusnet_total_params))
    return defocusnet

def load_RAFT(TRAIN_PARAMS, OUTPUT_PARAMS):
    raft = importlib.import_module('RAFT.raft').RAFT(TRAIN_PARAMS['RAFT_SET']['SMALL'], TRAIN_PARAMS['RAFT_SET']['ITERS'])
    if TRAIN_PARAMS['RAFT_LOAD']:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['RAFT_NAME']+'/'+TRAIN_PARAMS['RAFT_NAME']+'_ep' + str(TRAIN_PARAMS['RAFT_EP']) + '.pth'
        print("load model:", model)
        raft.load_state_dict(torch.load(model)['raft'], strict=False)
        #freeze
        if TRAIN_PARAMS['RAFT_FREEZE']:
            for name, value in raft.named_parameters():
                value.requires_grad = False
    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'/'+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'
        raft.load_state_dict(torch.load(model)['raft'])
        
    raft_total_params = sum(p.numel() for p in raft.parameters())
    raft_total_params_train = sum(p.numel() for p in raft.parameters() if p.requires_grad)
    print("RAFT's Total number of trainable params/Total number:",
          str(raft_total_params_train) + "/" + str(raft_total_params))
    return raft

def load_FusionNet(TRAIN_PARAMS, DATA_PARAMS, OUTPUT_PARAMS):
    fusion = importlib.import_module('FusionNet_nodefocus').FusionNet(stack_num = DATA_PARAMS['FOCUS_NUM'], 
                                                            res_blks = TRAIN_PARAMS['FUSIONNET_SET']['RES_BLOCKS_NUM'], 
                                                            channel = TRAIN_PARAMS['FUSIONNET_SET']['CHANNEL'], 
                                                            ref_block_size = TRAIN_PARAMS['FUSIONNET_SET']['REF_BLOCKS_SIZE'],
                                                            search_size = TRAIN_PARAMS['FUSIONNET_SET']['SEARCH_SIZE'],
                                                            stride = TRAIN_PARAMS['FUSIONNET_SET']['STRIDE'])
    if TRAIN_PARAMS['FUSION_LOAD']:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['FUSION_NAME']+'/'+TRAIN_PARAMS['FUSION_NAME']+'_ep' + str(TRAIN_PARAMS['FUSION_EP']) + '.pth'
        print("load model:", model)
        fusion.load_state_dict(torch.load(model), strict=False)
    fusion_total_params = sum(p.numel() for p in fusion.parameters())
    fusion_total_params_train = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print("FusionNet's Total number of trainable params/Total number:",
          str(fusion_total_params_train) + "/" + str(fusion_total_params))
    if TRAIN_PARAMS['EPOCH_START'] > 0:
        model = OUTPUT_PARAMS['MODEL_PATH']+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'/'+TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)+'_ep' + str(TRAIN_PARAMS['EPOCH_START']) + '.pth'
        fusion.load_state_dict(torch.load(model)['fusion'])
        print("Model loaded:", TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2), " epoch:", str(TRAIN_PARAMS['EPOCH_START']))
    return fusion

def set_comp_device(DEVICE):
    if DEVICE == 'CPU':
        device_comp = torch.device("cpu")
    elif DEVICE == 'GPU0':
        device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif DEVICE == 'GPU1':
        device_comp = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    return device_comp

def set_output_folders(OUTPUT_PARAMS, TRAIN_PARAMS):
    model_name = TRAIN_PARAMS['MODEL'] + str(OUTPUT_PARAMS['EXP_NUM']).zfill(2)
    res_dir = OUTPUT_PARAMS['RESULT_PATH'] + model_name + '/'
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + model_name + '/'
    if not isdir(models_dir):
        mkdir(models_dir)
    if not isdir(res_dir):
        mkdir(res_dir)
    return models_dir, model_name, res_dir

def save_config(r):
    model_name = r['TRAIN_PARAMS']['MODEL'] + str(r['OUTPUT_PARAMS']['EXP_NUM']).zfill(2)

    with open(r['OUTPUT_PARAMS']['MODEL_PATH'] + model_name+'/'+ 'configs_' + model_name + '.csv', mode='w', newline='') as res_file:
        res_writer = csv.writer(res_file, dialect=("excel"))
        res_writer.writerow(['model name:', model_name])
        res_writer.writerow(r['TRAIN_PARAMS'].keys())
        res_writer.writerow(r['TRAIN_PARAMS'].values())
        res_writer.writerow(r['DATA_PARAMS'].keys())
        res_writer.writerow(r['DATA_PARAMS'].values())
        res_writer.writerow(r['OUTPUT_PARAMS'].keys())
        res_writer.writerow(r['OUTPUT_PARAMS'].values())

# Visualize current progress
class Visualization():
    def __init__(self, port, hostname, model_name, flag_show=['defocus', 'depth'], env_name='main'):
        self.viz = Visdom(port=port, server=hostname, env=env_name)
        self.loss_plot = self.viz.line(X=[0.], Y=[0.], name="train", opts=dict(title='Loss ' + model_name))
        self.flag_show = flag_show

    def initial_viz(self, loss_val, viz_show):
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        self.viz.line(Y=[loss_val], X=[0], win=self.loss_plot, name="train", update='replace')
        self.img_show = []
        for i in range(len(viz_show)):
            viz_img = torch.clamp(viz_show[i], 0., 1.)
            if viz_show[i].shape[1] > 3 or viz_show[i].shape[1] == 2:
                viz_img = viz_img[:, 0:1, :, :]
            self.img_show.append(self.viz.images(viz_img, nrow=3, opts=dict(title=self.flag_show[i])))

    def log_viz_img(self, viz_show):
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        for i in range(len(viz_show)):
            viz_img = torch.clamp(viz_show[i], 0., 1.)
            if viz_show[i].shape[1] > 3 or viz_show[i].shape[1] == 2:
                viz_img = viz_img[:, 0:1, :, :]
            self.img_show[i] = self.viz.images(viz_img, win=self.img_show[i], nrow=3, opts=dict(title=self.flag_show[i]))

    def log_viz_plot(self, loss_val, total_iter):
        self.viz.line(Y=[loss_val], X=[total_iter], win=self.loss_plot, name="train", update='append')

