# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:22:09 2021

@author: Ming
"""
import torch
import torch.optim as optim
import torch.utils.data
from sacred import Experiment
from os import mkdir
from os.path import isdir
import kornia
import time
from torch.cuda.amp import GradScaler, autocast
import random

from utils_func import *
from losses import *
from DataLoader import *

exp = Experiment()

@exp.config
def my_config():
    TRAIN_PARAMS = {
        'LEARNING_RATE': 0.0001,
        'WEIGHT_DECAY': 0.00005,
        'EPSILON': 1e-8,
        'DEFOCUSNET_SET':{
            'FILTER_NUM': 32,
            'BLOCKS_NUM': 2,
            },
        'DEVICE': 'GPU0',#choose CPU,GPU0,GPU1
        'EPOCHS_NUM': 30, 'EPOCH_START': 0,
        
        'MODEL':'DEFOCUS',
        
        'DEFOCUS_LOAD': False,
        'DEFOCUS_NAME': "DEFOCUS01",
        'DEFOCUS_EP': 5,
        
        'RAFT_LOAD': False,
        'RAFT_NAME': "RAFT00",
        'RAFT_EP': 20,
        'RAFT_FREEZE':True,
        
        'FUSION_LOAD': False,
        'FUSION_NAME': "RAFT00",
        'FUSION_EP': 20,
        
        'DEFOCUS_L1_LOSS_WEIGHT':10,
        'DEFOCUS_SOBEL_LOSS_WEIGHT':10,
        'DEFOCUS_TV_LOSS_WEIGHT':0.0001,
    }

    DATA_PARAMS = {
        'DATA_PATH': './dataset/',
        'FLAG_TO_DATA': {
            'AIF': False,
            'DEFOCUS': True,
            'DEPTH': False,
            'MASK' : False,
            'INFO': False,
        },
        'FOCUS_NUM': 5,
        'TRAIN_SPLIT': 1,
        'DATASET_SHUFFLE': True,
        'DATA_ENHANCE':False,
        'WORKERS_NUM': 4,
        'BATCH_SIZE': 5,
    }

    OUTPUT_PARAMS = {
        'RESULT_PATH': './results/',
        'MODEL_PATH': './models/',
        'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'defocus',
        'VIZ_SHOW':['input1', 'defocus1', 'GT1',
                    'input2', 'defocus2', 'GT2',
                    'input3', 'defocus3', 'GT3',
                    'input4', 'defocus4', 'GT4',
                    'input5', 'defocus5', 'GT5'],
        'EXP_NUM': 3,
        'VIZ_UPDATE': 10,
        'NOTE':'defocus'
    }

load_defocusnet = exp.capture(load_defocusnet)
load_raft = exp.capture(load_RAFT)
load_fusion = exp.capture(load_FusionNet)
load_data = exp.capture(load_data, prefix='DATA_PARAMS')
set_comp_device = exp.capture(set_comp_device, prefix='TRAIN_PARAMS')
set_output_folders = exp.capture(set_output_folders)

@exp.capture
def train_defocus_model(loaders, model_info, viz_info, TRAIN_PARAMS, DATA_PARAMS, OUTPUT_PARAMS):
    model_info['defocusnet'].train()
    defocusnet_params = model_info['defocusnet'].parameters()
    defocusnet_params = filter(lambda p: p.requires_grad, defocusnet_params)
    L1 = torch.nn.L1Loss()
    optimizer = optim.AdamW(defocusnet_params, lr=TRAIN_PARAMS['LEARNING_RATE'], weight_decay=TRAIN_PARAMS['WEIGHT_DECAY'], eps=TRAIN_PARAMS['EPSILON'])
    scaler = GradScaler(enabled=True)
    
    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        start = time.time()
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count = 0, 0
        L1_loss_defocus_sum, SOBEL_loss_defocus_sum,TV_loss_defocus_sum =0,0,0
        
        for st_iter, sample_batch in enumerate(loaders[0]):
            X = []
            GT = []
            for index in range(DATA_PARAMS['FOCUS_NUM']):
                X.append(sample_batch["input{0}".format(index)].to(model_info['device_comp']))
                GT.append(sample_batch["defocus{0}".format(index)].to(model_info['device_comp']))
            optimizer.zero_grad()
            
            with autocast():
                # Forward 
                defocus = model_info['defocusnet'](X)
            
                # compute loss
                for i in range(DATA_PARAMS['FOCUS_NUM']):
                    if i == 0:
                        L1_loss_defocus = L1(defocus[i], GT[i])
                        SOBEL_loss_defocus = L1(kornia.sobel(defocus[i]),kornia.sobel(GT[i]))
                        TV_loss = kornia.total_variation(defocus[i])
                        TV_loss_defocus = sum(TV_loss) / len(TV_loss)
                    else:
                        L1_loss_defocus += L1(defocus[i], GT[i])
                        SOBEL_loss_defocus += L1(kornia.sobel(defocus[i]),kornia.sobel(GT[i]))
                        TV_loss = kornia.total_variation(defocus[i])
                        TV_loss_defocus += sum(TV_loss) / len(TV_loss)
               
                L1_loss_defocus = L1_loss_defocus * TRAIN_PARAMS['DEFOCUS_L1_LOSS_WEIGHT']
                SOBEL_loss_defocus = SOBEL_loss_defocus* TRAIN_PARAMS['DEFOCUS_SOBEL_LOSS_WEIGHT']
                TV_loss_defocus = TV_loss_defocus * TRAIN_PARAMS['DEFOCUS_TV_LOSS_WEIGHT']
                    
                loss = L1_loss_defocus + SOBEL_loss_defocus + TV_loss_defocus
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            # Training log
            L1_loss_defocus_sum += L1_loss_defocus.item()
            SOBEL_loss_defocus_sum += SOBEL_loss_defocus.item()
            TV_loss_defocus_sum += TV_loss_defocus.item()
            loss_sum += loss.item()
            iter_count += 1.

            if (st_iter + 1) % OUTPUT_PARAMS['VIZ_UPDATE'] == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}]: \n \
                      Defocus L1Loss: {:.4f}, Defocus Sobel Loss: {:.4f}, Defocus TVLoss: {:.4f}, Total Loss: {:.4f}\n'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], 
                              L1_loss_defocus_sum / iter_count, SOBEL_loss_defocus_sum / iter_count, 
                              TV_loss_defocus_sum / iter_count, loss_sum / iter_count))
                total_iter = model_info['total_steps'] * epoch_iter + st_iter + 1
                if epoch_iter == TRAIN_PARAMS['EPOCH_START'] and (st_iter + 1) == OUTPUT_PARAMS['VIZ_UPDATE']:
                    viz_info.initial_viz(loss_val=loss_sum / iter_count,
                                         viz_show=[X[0].detach(), defocus[0].detach(), GT[0].detach(),
                                                   X[1].detach(), defocus[1].detach(), GT[1].detach(),
                                                   X[2].detach(), defocus[2].detach(), GT[2].detach(),
                                                   X[3].detach(), defocus[3].detach(), GT[3].detach(),
                                                   X[4].detach(), defocus[4].detach(), GT[4].detach()])
                else:
                    viz_info.log_viz_plot(loss_val=loss_sum / iter_count, total_iter=total_iter)
                    viz_info.log_viz_img(viz_show=[X[0].detach(), defocus[0].detach(), GT[0].detach(),
                                                   X[1].detach(), defocus[1].detach(), GT[1].detach(),
                                                   X[2].detach(), defocus[2].detach(), GT[2].detach(),
                                                   X[3].detach(), defocus[3].detach(), GT[3].detach(),
                                                   X[4].detach(), defocus[4].detach(), GT[4].detach()])
                loss_sum, iter_count = 0, 0
                L1_loss_defocus_sum, SOBEL_loss_defocus_sum,TV_loss_defocus_sum= 0,0,0

        # Save model
        state = {'defocusnet':model_info['defocusnet'].state_dict()}
        torch.save(state, model_info['model_dir'] + model_info['model_name'] + '_ep' + str(epoch_iter + 1) + '.pth')
        end = time.time()
        running_time = end-start
        print('time cost : %.5f sec' %running_time)
        
@exp.capture
def train_raft_model(loaders, model_info, viz_info, TRAIN_PARAMS, DATA_PARAMS, OUTPUT_PARAMS):
    model_info['raft'].train()
    raft_params = model_info['raft'].parameters()
    raft_params = filter(lambda p: p.requires_grad, raft_params)
    optimizer = optim.AdamW(raft_params, lr=TRAIN_PARAMS['LEARNING_RATE'], weight_decay=TRAIN_PARAMS['WEIGHT_DECAY'], eps=TRAIN_PARAMS['EPSILON'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, TRAIN_PARAMS['LEARNING_RATE'], TRAIN_PARAMS['EPOCHS_NUM']*model_info['total_steps']+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scaler = GradScaler(enabled=True)
    
    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        start = time.time()
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count = 0, 0

        for st_iter, sample_batch in enumerate(loaders[0]):
            optimizer.zero_grad()
            index = random.sample(range(0, DATA_PARAMS['FOCUS_NUM']), 2)
            image1 = sample_batch["input{0}".format(index[0])].to(model_info['device_comp'])
            image2 = sample_batch["input{0}".format(index[1])].to(model_info['device_comp'])
            K = sample_batch['info']['K'].to(model_info['device_comp'])
            RT1 = sample_batch['info']["RT{0}".format(index[0])].to(model_info['device_comp'])
            RT2 = sample_batch['info']["RT{0}".format(index[1])].to(model_info['device_comp'])
            flow = compute_flow(sample_batch["depth{0}".format(index[0])].to(model_info['device_comp']), K, RT1, RT2)
            # Forward
            flow_predictions = model_info['raft'](image1, image2)

            # compute loss
            loss = sequence_loss(flow_predictions, flow)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raft_params, 1.0)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            # Training log
            loss_sum += loss.item()
            iter_count += 1.

            if (st_iter + 1) % OUTPUT_PARAMS['VIZ_UPDATE'] == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}]: Loss: {:.4f}\n'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], loss_sum / iter_count))
                total_iter = model_info['total_steps'] * epoch_iter + st_iter + 1
                if epoch_iter == TRAIN_PARAMS['EPOCH_START'] and (st_iter + 1) == OUTPUT_PARAMS['VIZ_UPDATE']:
                    warp_img_pre = warp(image2.detach(), flow_predictions[-1].detach())
                    warp_img_gt = warp(image2.detach(), flow.detach())
                    viz_info.initial_viz(loss_val=loss_sum / iter_count,
                                          viz_show=[image1.detach(), image2.detach(), warp_img_pre.detach(), warp_img_gt.detach()])
                else:
                    warp_img_pre = warp(image2.detach(), flow_predictions[-1].detach())
                    warp_img_gt = warp(image2.detach(), flow.detach())
                    viz_info.log_viz_plot(loss_val=loss_sum / iter_count, total_iter=total_iter)
                    viz_info.log_viz_img(viz_show=[image1.detach(), image2.detach(), warp_img_pre.detach(), warp_img_gt.detach()])
                loss_sum, iter_count = 0, 0

        # Save model
        state = {'raft':model_info['raft'].state_dict()}
        torch.save(state, model_info['model_dir'] + model_info['model_name'] + '_ep' + str(epoch_iter + 1) + '.pth')
        end = time.time()
        running_time = end-start
        print('time cost : %.5f sec' %running_time)
        
@exp.capture
def train_fusion_model(loaders, model_info, viz_info, TRAIN_PARAMS, DATA_PARAMS, OUTPUT_PARAMS):
    model_info['fusionnet'].train()
    fusion_params = model_info['fusionnet'].parameters()
    fusion_params = filter(lambda p: p.requires_grad, fusion_params)
    
    if TRAIN_PARAMS['DEFOCUS_FREEZE'] == True and TRAIN_PARAMS['RAFT_FREEZE'] == True:
        model_info['defocusnet'].eval()
        model_info['raft'].eval()
        optimizer = optim.Adam(fusion_params, lr=TRAIN_PARAMS['LEARNING_RATE'], weight_decay=TRAIN_PARAMS['WEIGHT_DECAY'])
    else:
        model_info['defocusnet'].train()
        model_info['raft'].train()
        defocusnet_params = model_info['defocusnet'].parameters()
        defocusnet_params = filter(lambda p: p.requires_grad, defocusnet_params)
        raft_params = model_info['raft'].parameters()
        raft_params = filter(lambda p: p.requires_grad, raft_params)
        optimizer = optim.Adam([
                {'params': defocusnet_params, 'lr': TRAIN_PARAMS['DEFOCUSNET_SET']['LEARNING_RATE']},
                {'params': raft_params, 'lr': TRAIN_PARAMS['RAFT_SET']['LEARNING_RATE']},
                {'params': fusion_params, 'lr': TRAIN_PARAMS['LEARNING_RATE']},
                ], weight_decay=TRAIN_PARAMS['WEIGHT_DECAY'])
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAIN_PARAMS['EPOCHS_NUM']*model_info['total_steps']+100)
    scaler = GradScaler(enabled=True)
    
    ##### Training
    print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
    
    L1 = torch.nn.L1Loss()
    for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
        start = time.time()
        epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
        loss_sum, iter_count = 0, 0
        L1_loss_sum, VGG_loss_sum, SSIM_loss_sum = 0,0,0

        for st_iter, sample_batch in enumerate(loaders[0]):
            optimizer.zero_grad()
            # data prepare
            ref_index = random.randint(0, DATA_PARAMS['FOCUS_NUM']-1)
            img = []
            X = []
            GT = sample_batch["aif{0}".format(ref_index)].to(model_info['device_comp'])
            for i in range(DATA_PARAMS['FOCUS_NUM']):
                X.append(sample_batch["input{0}".format(i)].to(model_info['device_comp']))
            defocus = model_info['defocusnet'](X)
            for i in range(DATA_PARAMS['FOCUS_NUM']):
                if i != ref_index:
                    flow = model_info['raft'](sample_batch["input{0}".format(ref_index)].to(model_info['device_comp']), 
                                              sample_batch["input{0}".format(i)].to(model_info['device_comp']))
                    img.append(warp(X[i], flow[-1]))
                    defocus[i] = warp((1-defocus[i]),flow[-1])
                else:
                    img.append(sample_batch["input{0}".format(i)].to(model_info['device_comp']))
                    defocus[i] = 1-defocus[i]
            with autocast():
                # Forward
                pre = model_info['fusionnet'](img, defocus, ref_index)
                # compute loss
                L1_loss = L1(pre, GT) * TRAIN_PARAMS['RECONSTRUCT_LOSS_WEIGHT']
                VGG_loss = perceptualLoss(pre, GT) * TRAIN_PARAMS['PERCEPTUAL_LOSS_WEIGHT']
                ssim_loss = kornia.losses.ssim_loss(pre, GT, 5) * TRAIN_PARAMS['SSIM_LOSS_WEIGHT']
                loss = L1_loss + VGG_loss + ssim_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            # Training log
            L1_loss_sum += L1_loss.item()
            VGG_loss_sum += VGG_loss.item()
            SSIM_loss_sum += ssim_loss.item()
            loss_sum += loss.item()
            iter_count += 1.

            if (st_iter + 1) % OUTPUT_PARAMS['VIZ_UPDATE'] == 0:
                print(model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}]: \n \
                      Reconstruction Loss: {:.4f}, Perceptual Loss: {:.4f}, SSIM Loss: {:.4f}, Total Loss: {:.4f}\n'
                      .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, model_info['total_steps'], 
                              L1_loss_sum / iter_count, VGG_loss_sum / iter_count, 
                              SSIM_loss_sum / iter_count, loss_sum / iter_count))
                total_iter = model_info['total_steps'] * epoch_iter + st_iter + 1
                if epoch_iter == TRAIN_PARAMS['EPOCH_START'] and (st_iter + 1) == OUTPUT_PARAMS['VIZ_UPDATE']:
                    viz_info.initial_viz(loss_val=loss_sum / iter_count,
                                         viz_show=[X[0].detach(), X[1].detach(),
                                                   X[2].detach(), X[3].detach(),
                                                   X[4].detach(), pre.detach(), GT.detach()])
                else:
                    viz_info.log_viz_plot(loss_val=loss_sum / iter_count, total_iter=total_iter)
                    viz_info.log_viz_img(viz_show=[X[0].detach(), X[1].detach(),
                                                   X[2].detach(), X[3].detach(),
                                                   X[4].detach(), pre.detach(), GT.detach()])
                loss_sum, iter_count = 0, 0
                L1_loss_sum, VGG_loss_sum,SSIM_loss_sum = 0,0,0

        # Save model
        scheduler.step()
        state = {'defocusnet':model_info['defocusnet'].state_dict(),
                 'raft':model_info['raft'].state_dict(),
                 'fusion':model_info['fusionnet'].state_dict()}
        torch.save(state, model_info['model_dir'] + model_info['model_name'] + '_ep' + str(epoch_iter + 1) + '.pth')
        end = time.time()
        running_time = end-start
        print('time cost : %.5f sec' %running_time)

@exp.capture
def test_model(loaders, model_info, i, TRAIN_PARAMS, DATA_PARAMS):
    test_model_name = model_info['model_dir'] + model_info['model_name'] + '_ep' + str(i) + '.pth'
    print('test model:',test_model_name)
    test_dict = torch.load(test_model_name)
    iter_count = 0
    img_save_path1 = model_info['result_dir']
    if not isdir(img_save_path1):
        mkdir(img_save_path1)

    model_info['defocusnet'].load_state_dict(test_dict['defocusnet'])
    model_info['defocusnet'].eval()
    model_info['raft'].load_state_dict(test_dict['raft'])
    model_info['raft'].eval()
    model_info['fusionnet'].load_state_dict(test_dict['fusion'])
    model_info['fusionnet'].eval()

    with torch.no_grad():
        for st_iter, sample_batch in enumerate(loaders[2]):
            print(st_iter)
            # data prepare
            img_save_path = img_save_path1 + str(st_iter) + '/'
            if not isdir(img_save_path):
                mkdir(img_save_path)
            ref_index = 2
            img = []
            defocus = []
            GT = sample_batch["aif{0}".format(ref_index)].to(model_info['device_comp'])
            
            for i in range(DATA_PARAMS['FOCUS_NUM']):
                img.append(sample_batch["input{0}".format(i)].to(model_info['device_comp']))
            defocus = model_info['defocusnet'](img)
            for i in range(DATA_PARAMS['FOCUS_NUM']):
                if i != ref_index:
                    flow = model_info['raft'](img[ref_index], img[i])
                    img[i]=warp(img[i], flow[-1])
                    defocus[i] = warp((1-defocus[i]),flow[-1])
                else:
                    defocus[i] = 1-defocus[i]
            
            pre = model_info['fusionnet'](img, defocus, ref_index)
            
            GT = tensor_to_img(GT)
            GT.save(img_save_path +'GT.tif')
            pre = tensor_to_img(pre)
            pre.save(img_save_path +'Predict.tif')
            iter_count += 1.

@exp.automain
def run_exp(TRAIN_PARAMS, OUTPUT_PARAMS, DATA_PARAMS):
    # Initial preparations
    model_dir, model_name, res_dir = set_output_folders()
    device_comp = set_comp_device()
    print('device:',device_comp)

    # Training initializations
    loaders, total_steps = load_data()
    defocusnet = None
    raft = None
    fusionnet = None

    if TRAIN_PARAMS['MODEL'] == 'DEFOCUS' or TRAIN_PARAMS['DEFOCUS_LOAD']:
        defocusnet = load_defocusnet()
        defocusnet = defocusnet.to(device=device_comp)
    if TRAIN_PARAMS['MODEL'] == 'RAFT' or TRAIN_PARAMS['RAFT_LOAD']:
        raft = load_raft()
        raft = raft.to(device=device_comp)
    if TRAIN_PARAMS['MODEL'] == 'FUSION' or TRAIN_PARAMS['FUSION_LOAD']:
        fusionnet = load_fusion()
        fusionnet = fusionnet.to(device=device_comp)

    model_info = {'defocusnet': defocusnet,
                  'raft': raft,
                  'fusionnet':fusionnet,
                  'model_dir': model_dir,
                  'result_dir': res_dir,
                  'model_name': model_name,
                  'total_steps': total_steps,
                  'device_comp': device_comp,
                  }
    save_config(my_config())

    # set visualization
    viz_info = Visualization(OUTPUT_PARAMS['VIZ_PORT'], OUTPUT_PARAMS['VIZ_HOSTNAME'], model_name, 
                             OUTPUT_PARAMS['VIZ_SHOW'],
                             env_name=OUTPUT_PARAMS['VIZ_ENV_NAME'])
    # Run training
    if TRAIN_PARAMS['MODEL'] == 'DEFOCUS':
        train_defocus_model(loaders=loaders, model_info=model_info, viz_info=viz_info)
    if TRAIN_PARAMS['MODEL'] == 'RAFT':
        train_raft_model(loaders=loaders, model_info=model_info, viz_info=viz_info)
    if TRAIN_PARAMS['MODEL'] == 'FUSION':
        train_fusion_model(loaders=loaders, model_info=model_info, viz_info=viz_info)
        test_model(loaders, model_info, 20, TRAIN_PARAMS, DATA_PARAMS)


