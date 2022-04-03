# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:10:10 2021

@author: Ming
"""
import torch
import torch.utils.data
from torchvision import transforms
from os import listdir
from PIL import Image
import numpy as np
import csv
import random

class StackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scene_num, foc_num, 
                 flag_outputs={'AIF':False,'DEFOCUS':False,'DEPTH':False,'MASK':False,'INFO':False,}, transform_fnc=None):
        self.transform_fnc = transform_fnc
        self.foc_num = foc_num
        self.flag_out_aif = flag_outputs['AIF']
        self.flag_out_defocus = flag_outputs['DEFOCUS']
        self.flag_out_depth = flag_outputs['DEPTH']
        self.flag_out_mask = flag_outputs['MASK']
        self.flag_out_info = flag_outputs['INFO']
        self.dic={}
        for i in range(foc_num):
            self.dic["input{0}".format(i)]=[]
            self.dic["aif_img{0}".format(i)]=[]
            self.dic["defocus_map{0}".format(i)]=[]
            self.dic["depth_map{0}".format(i)]=[]
            self.dic["mask_img{0}".format(i)]=[]
            self.dic["info_csv{0}".format(i)]=[]

        for i in scene_num:
            for j in range(foc_num):
                self.dic["input{0}".format(j)].append(root_dir+str(i)+"/defocus_"+str(j)+".tif")
                self.dic["aif_img{0}".format(j)].append(root_dir+str(i)+"/all_in_focus_"+str(j)+".tif")
                self.dic["defocus_map{0}".format(j)].append(root_dir+str(i)+"/defocusmap_"+str(j)+".exr")
                self.dic["depth_map{0}".format(j)].append(root_dir+str(i)+"/depth_"+str(j)+".exr")
                self.dic["mask_img{0}".format(j)].append(root_dir+str(i)+"/mask_"+str(j)+".png")
                self.dic["info_csv{0}".format(j)].append(root_dir+str(i)+"/info_"+str(j)+".csv")


    def __len__(self):
        return len(self.dic['aif_img0'])

    def __getitem__(self, index):
        ##### Read and process an image
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        sample = {}
        sample['info'] = {}
        for i in range(self.foc_num):
            #source image
            inp = Image.open(self.dic["input{0}".format(i)][index])
            inp = np.array(inp).astype(np.float32)
            sample["input{0}".format(i)] = inp/255.0
            #all-in-focus
            if self.flag_out_aif:
                aif = Image.open(self.dic["aif_img{0}".format(i)][index])
                aif = np.array(aif).astype(np.float32)
                sample["aif{0}".format(i)] = aif/255.0
            #depth
            if self.flag_out_depth:
                depth = cv2.imread(self.dic["depth_map{0}".format(i)][index], cv2.IMREAD_UNCHANGED)
                depth = depth[:,:,0]
                depth = depth[:, :, np.newaxis]
                sample["depth{0}".format(i)] = depth
            #defocus
            if self.flag_out_defocus:
                defocus_min=[]
                defocus_max=[]
                for k in range(self.foc_num):
                    defocus = cv2.imread(self.dic["defocus_map{0}".format(k)][index], cv2.IMREAD_UNCHANGED)
                    defocus_min.append(defocus.min())
                    defocus_max.append(defocus.max())
                defocus = cv2.imread(self.dic["defocus_map{0}".format(i)][index], cv2.IMREAD_UNCHANGED)
                defocus = defocus[:, :, np.newaxis]
                defocus = (defocus - min(defocus_min)) / (max(defocus_max)-min(defocus_min))
                sample["defocus{0}".format(i)] = defocus
            #mask
            if self.flag_out_mask:
                mask = Image.open(self.dic["mask_img{0}".format(i)][index])
                mask = np.array(mask).astype(np.float32)
                mask = mask[:,:, np.newaxis]
                sample["mask{0}".format(i)] = mask/mask.max()
            #information
            if self.flag_out_info:
                with open(self.dic["info_csv{0}".format(i)][index], mode='r') as inp:
                    reader = csv.reader(inp)
                    info = {rows[0]:eval(rows[1]) for rows in reader}
                K = info['K']
                K = np.array(K, dtype=np.float32)
                sample['info']['K'] = torch.from_numpy(K)
                RT = info['RT']
                RT = np.array(RT, dtype=np.float32)
                sample['info']["RT{0}".format(i)] = torch.from_numpy(RT)

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        return sample
    
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for i in sample.keys():
                if i != 'info':
                    img = sample[i]
                    img = Image.fromarray(img)
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img = np.array(img, dtype=np.float32) 
                    sample[i] = img
        return sample

class RandomVerticalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for i in sample.keys():
                if i != 'info':
                    img = sample[i]
                    img = Image.fromarray(img)
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img = np.array(img, dtype=np.float32) 
                    sample[i] = img
        return sample


class ToTensor(object):
    def __call__(self, sample):
        for i in sample.keys():
            if i != 'info':
                img = sample[i]
                img = img.transpose((2, 0, 1))
                img = torch.from_numpy(img).float()
                sample[i] = img
        return sample
    
def load_data(DATA_PATH, TEST_PATH, FLAG_TO_DATA, FOCUS_NUM, TRAIN_SPLIT,
              DATASET_SHUFFLE, DATA_ENHANCE, WORKERS_NUM, BATCH_SIZE):
    #dataset split
    train_num = random.sample(range(len(listdir(DATA_PATH))), int(len(listdir(DATA_PATH)) * TRAIN_SPLIT))
    valid_num = []
    for item in range(len(listdir(DATA_PATH))):
        if item not in train_num:
            valid_num.append(item)
    test_num = []
    for item in range(len(listdir(TEST_PATH))):
        test_num.append(item)
    if DATA_ENHANCE:
        transform_fnc=transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),ToTensor()])
    else:
        transform_fnc=transforms.Compose([ToTensor()])
    
    train_dataset = StackDataset(root_dir=DATA_PATH, scene_num = train_num,foc_num = FOCUS_NUM,flag_outputs=FLAG_TO_DATA, transform_fnc=transform_fnc)
    valid_dataset = StackDataset(root_dir=DATA_PATH, scene_num = valid_num,foc_num = FOCUS_NUM,flag_outputs=FLAG_TO_DATA,
                           transform_fnc=transforms.Compose([ToTensor()]))
    test_dataset = StackDataset(root_dir=TEST_PATH, scene_num = test_num,foc_num = FOCUS_NUM,flag_outputs=FLAG_TO_DATA,
                           transform_fnc=transforms.Compose([ToTensor()]))

    if DATASET_SHUFFLE:
        indices_train = random.sample(range(len(train_dataset)), len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, indices_train)

    loader_train = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset=valid_dataset, num_workers=WORKERS_NUM, batch_size=1, shuffle=False)
    loader_test = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=WORKERS_NUM, batch_size=1, shuffle=False)
    total_steps = int(len(train_dataset) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(train_dataset))
    print("Total number of testing sample:", len(test_dataset))

    return [loader_train, loader_valid, loader_test], total_steps