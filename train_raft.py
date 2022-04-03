# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:43:52 2021

@author: Ming
"""
from train import exp

# default parameters
if True:
    r = exp.run(config_updates={
        'TRAIN_PARAMS': {
            'LEARNING_RATE': 0.0001,
            'WEIGHT_DECAY': 0.00005,
            'EPSILON': 1e-8,
            'RAFT_SET':{
            'ITERS': 10,
            'SMALL': True},
            
            'DEVICE': 'GPU0',#choose CPU,GPU0,GPU1
            'EPOCHS_NUM': 100, 'EPOCH_START': 0,
        
            'MODEL':'RAFT',
        
            'RAFT_LOAD': False,
            'RAFT_NAME': "DEFOCUS01",
            'RAFT_EP': 5,
                        },
        'DATA_PARAMS': {
            'DATA_PATH': './dataset/',
            'FLAG_TO_DATA': {
                'AIF': False,
                'DEFOCUS': False,
                'DEPTH': True,
                'MASK' : False,
                'INFO': True,
                },
            'FOCUS_NUM': 5,
            'TRAIN_SPLIT': 1,
            'DATASET_SHUFFLE': True,
            'DATA_ENHANCE':False,
            'WORKERS_NUM': 4,
            'BATCH_SIZE': 15,
                        },
        'OUTPUT_PARAMS': {
            'RESULT_PATH': './results/',
            'MODEL_PATH': './models/',
            'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'raft',
            'VIZ_SHOW':['input1','input2','pre', 'GT'],
            'EXP_NUM': 0,
            'VIZ_UPDATE': 10,
            'NOTE':'optical flow estimate'
        }
        })