#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:18:59 2020

@author: trivizakis
"""
import os
import sys
import cv2
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plot
from skimage.transform import resize

import tensorflow as tf

sys.path.append('../unet')
from model import unet

def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., np.newaxis]
    return pred_mask[0]


exp_dir = "lung_deep_segmentation"

params={}
with open("chkp/"+exp_dir+"/logs/hypes",'r') as file:
    params = json.load(file)
    
segmentation_model = unet(input_size=params["input_shape"],n_filters=params["num_filters"], activation=params["activation"], num_classes=params["num_classes"])

segmentation_model.load_weights("chkp/"+exp_dir+"/model--chpoint-03-0.51.h5")

for layer in segmentation_model.layers:
    layer.trainable = False
    
for _, _, patients in os.walk("dataset/npy_tosegment"):
    for patient in patients:
        print(patient)
        original_slice = np.load("dataset/npy_tosegment/"+patient)            
        if original_slice.shape[0]!=512:
            print("Resizing: "+patient)
            ct_slice = resize(original_slice, (512,512))
        else:
            ct_slice = original_slice
        img = np.expand_dims(np.expand_dims((ct_slice/255.), axis=0), axis=3)
        prd_msk = segmentation_model.predict(img)
        msk = create_mask(prd_msk).astype(np.uint8)[:,:,0]
        plot.imsave("dataset/predictions/"+patient+"_MASK.png", msk*255,cmap="gray")
        plot.imsave("dataset/predictions/"+patient+".png", ct_slice,cmap="gray")
