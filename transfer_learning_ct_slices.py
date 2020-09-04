#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import pandas as pd
import numpy as np
import pickle as pkl
import os
import cv2
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D, MaxPooling2D, GlobalMaxPool2D
from keras.models import Model, Sequential
from keras.layers.core import Lambda

import sys
import itertools

from dataset_analysis import DatasetAnalysis
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.append('../cnn_framework')
from utils import Utils
from model import CustomModel
from dataset import DataConverter
from data_augmentation import DataAugmentation
from data_generator import DataGenerator

sys.path.append('../transfer_learning')
from deep_transfer_learning import Transferable_Networks

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    print(loss)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    print(conv_output)
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (512, 512))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def dataset_split(labels,hypes):
        skf_conv_tst = StratifiedKFold(n_splits=hypes["kfold"][0],shuffle=hypes["shuffle"])
        skf_tr_val = StratifiedShuffleSplit(n_splits=hypes["kfold"][1], test_size=0.2, train_size=0.8)
        values = np.array(list(labels.values()), dtype=int)
        pids =  np.array(list(labels.keys()), dtype=str)
        tr=[]
        val=[]
        tst=[]
        
        for trval_index, tst_index in skf_conv_tst.split(pids,values):
            tmp_tr=[]
            tmp_val=[]
            tmp_tst=[]
            convergence_pids = pids[trval_index]
            convergence_labels = values[trval_index]
            for tr_index, val_index in skf_tr_val.split(convergence_pids,convergence_labels):
                tmp_tr=convergence_pids[tr_index].tolist()
                tmp_val=convergence_pids[val_index].tolist()
                tmp_tst=pids[tst_index].tolist()
            tr.append(tmp_tr)
            val.append(tmp_val)
            tst.append(tmp_tst)
            
        return tr, val, tst
    
hypes = Utils.get_hypes("hypes2d")

with open("dataset/npy_analysis/labels.pkl",'rb') as file:
    classes = pkl.load(file)
    
training_set, validation_set, testing_set = dataset_split(classes,hypes)


best_performance={}
for index, training in enumerate(training_set):
    #network version
    version = str(index+1)+"."+str(1)
    hypes["version"] = "network_version:"+version+"/"
    
    #make dirs        
    Utils.make_dirs(version,hypes)    
    
    Utils.save_skf_pids(version+".dataset",np.array(training_set[index], dtype=str),np.array(validation_set[index], dtype=str),np.array(testing_set[index],dtype=str),hypes) #save pids
 
    
    # Generators
    training_generator = DataGenerator(training_set[index], classes, hypes, training=True)
    validation_generator = DataGenerator(validation_set[index], classes, hypes, training=False)
        
    #clear session in every iteration        
    K.clear_session()
    
    #full network
    include_top=False
    
    if hypes["avgpool"]:
        pooling = "avg"
    elif hypes["avgpool"] == "max":
        pooling = "max"
    else:
        pooling=None
    #create network
    tn = Transferable_Networks(hypes)
    pretrained = tn.get_pretrained(input_shape=tuple(hypes["input_shape"]), model_name="vgg", pooling=pooling, freeze_up_to=-1, include_top=include_top, classes=hypes["num_classes"])
    # inc3 incr2  - -22 two conv - -12 one conv, 299x299x3
    #small_densenet + medium_densenet + densenet- block15 -14, block 16 -10, 224x224
    #nasnet or mobile - -20, -35
    #vgg -  -10 2conv
    
    if not include_top:       
        if hypes["avgpool"]:
            net_out = pretrained.output
        else:
            net_out = Conv2D(
                    filters=256,
                    kernel_size=(3,3),
                    strides=(2,2),
                    padding=hypes["padding"],
                    name="conv_pre_in")(pretrained.output)
            net_out = Conv2D(
                    filters=256,
                    kernel_size=(3,3),
                    strides=(2,2),
                    padding=hypes["padding"],
                    name="conv_duo")(net_out)
            net_out = Flatten()(net_out)
        for neurons in hypes["neurons"]:
            net_out = Dense(units=neurons,activation="relu")(net_out)
            net_out = Dropout(hypes["dropout"])(net_out)
        classifier = Dense(units=hypes["num_classes"], activation=hypes["classifier"])(net_out)
        final_model = Model(inputs=pretrained.input, outputs=classifier)
    else:
        final_model = Model(inputs=pretrained.input, outputs=classifier)
    
    final_model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(lr=hypes["learning_rate"],decay=hypes["weight_decay"]),
                        metrics=hypes["metric"])
    print(final_model.summary())

    #fit network
    CustomModel.train_model(final_model,hypes,training_generator,validation_generator)

    
    #inference
    hypes_test = hypes
    hypes_test["batch_size"]=1
    testing_generator = DataGenerator(testing_set[index], classes, hypes_test, training=False)
    model_performance={}
    for filename in os.listdir(hypes["chkp_dir"]+hypes["version"]):
        if "weights" in filename:
            model_performance[int(float(filename[-7:-3])*100)]=filename
    final_model.load_weights(hypes["chkp_dir"]+hypes["version"]+model_performance[max(model_performance.keys())])
    _,_, roc, _ = CustomModel.test_model(final_model,hypes_test,testing_generator)
    print(filename+" : "+str(roc))
    best_performance[hypes["version"]+filename]=roc
    #save current hypes
    Utils.save_hypes(hypes["chkp_dir"]+hypes["version"], "hypes"+version, hypes)
