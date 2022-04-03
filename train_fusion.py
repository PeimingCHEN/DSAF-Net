# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:07:36 2021

@author: Ming
"""
from train import exp

# default parameters
if True:
    r = exp.run(config_updates={
        'TRAIN_PARAMS': {
            'LEARNING_RATE': 0.0001,
            'WEIGHT_DECAY': 0,
            
            'FUSIONNET_SET':{
                'RES_BLOCKS_NUM': 2,
                'CHANNEL': [8,16,32],
                'REF_BLOCKS_SIZE': [14, 6, 3],
                'SEARCH_SIZE': [20, 16, 8],
                'STRIDE': [6,4,1],
                },
            
            'DEFOCUSNET_SET':{
                'FILTER_NUM': 32,
                'BLOCKS_NUM': 2,
                'LEARNING_RATE': 0.00001,
                },
            
            'RAFT_SET':{
                'ITERS': 5,
                'SMALL': True,
                'LEARNING_RATE': 0.00001,
                },

            'DEVICE': 'GPU1',#choose CPU,GPU0,GPU1
            'EPOCHS_NUM': 20, 'EPOCH_START': 0,
        
            'MODEL':'FUSION',#choose DEFOCUS,RAFT,FUSION
            
            'DEFOCUS_LOAD': False,
            'DEFOCUS_NAME': "FUSION03",
            'DEFOCUS_EP': 20,
            'DEFOCUS_FREEZE': True,
            
            'RAFT_LOAD': True,
            'RAFT_NAME': "FUSION06",
            'RAFT_EP': 11,
            'RAFT_FREEZE':False,
            
            'FUSION_LOAD': False,
            'FUSION_NAME': "RAFT00",
            'FUSION_EP': 20,
            
            'RECONSTRUCT_LOSS_WEIGHT':10,
            'PERCEPTUAL_LOSS_WEIGHT':1,
            'SSIM_LOSS_WEIGHT':10,
                        },
        
        'DATA_PARAMS': {
            'DATA_PATH': './dataset/',
            'TEST_PATH': './test/',
            'FLAG_TO_DATA': {
                'AIF': True,
                'DEFOCUS': False,
                'DEPTH': False,
                'MASK' : False,
                'INFO': False,
                },
            'FOCUS_NUM': 5,
            'TRAIN_SPLIT': 1,
            'DATASET_SHUFFLE': True,
            'DATA_ENHANCE':False,
            'WORKERS_NUM': 4,
            'BATCH_SIZE': 3,
                        },
        'OUTPUT_PARAMS': {
            'RESULT_PATH': './results/',
            'MODEL_PATH': './models/',
            'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'fusion',
            'VIZ_SHOW':['input1','input2','input3','input4','input5','pre', 'GT'],
            'EXP_NUM': 6,
            'VIZ_UPDATE': 5,
        }
        })