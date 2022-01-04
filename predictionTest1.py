# -*- coding: utf-8 -*-

# IMPORT LIBRARIES:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import pandas as pd
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("Loading models...")
mball36 = tf.keras.models.load_model('models\egg36')
mball47 = tf.keras.models.load_model('models\egg47')
mball53 = tf.keras.models.load_model('models\egg53')
mball59 = tf.keras.models.load_model('models\egg59')
mball63 = tf.keras.models.load_model('models\egg63')
mball64 = tf.keras.models.load_model('models\egg64')
mball60 = tf.keras.models.load_model('models\egg60')
mball54 = tf.keras.models.load_model('models\egg54')
mball48 = tf.keras.models.load_model('models\egg48')
mball40 = tf.keras.models.load_model('models\egg40')
print("Models loaded...")

# Get start input data for model prediction:
rangeI = np.arange(0,96)
dataset_x = pd.read_csv('input.txt', usecols=rangeI, engine='python')
dataset_x = dataset_x.values
dataset_x = dataset_x.astype('float32')
x = np.reshape(dataset_x, (dataset_x.shape[0],1,dataset_x.shape[1]))

loopcnt = 2
for n in range(loopcnt):

    ball36 = mball36.predict(x)
    ball47 = mball47.predict(x)
    ball53 = mball53.predict(x)
    ball59 = mball59.predict(x)
    ball63 = mball63.predict(x)
    ball64 = mball64.predict(x)
    ball60 = mball60.predict(x)
    ball54 = mball54.predict(x)
    ball48 = mball48.predict(x)
    ball40 = mball40.predict(x)
    
    #preds = [int(round(ball36[ball36.shape[0]-1][0])),int(round(ball47[ball47.shape[0]-1][0])),int(round(ball53[ball53.shape[0]-1][0])),int(round(ball59[ball59.shape[0]-1][0])),int(round(ball63[ball63.shape[0]-1][0])),int(round(ball64[ball64.shape[0]-1][0])),int(round(ball60[ball60.shape[0]-1][0])),int(round(ball54[ball54.shape[0]-1][0])),int(round(ball48[ball48.shape[0]-1][0])),int(round(ball40[ball40.shape[0]-1][0]))]
    preds = []
    size = ball36.shape[0]-1
    thresh = 0.15
    
    if ball36[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball47[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball53[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball59[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball63[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball64[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball60[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball54[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball48[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    if ball40[size][0] > thresh:
        preds.append(1)
    else:
        preds.append(0)
        
    print(preds) 
    
    nxtState = x[x.shape[0]-1][0]
    nxtState[36+1] = preds[0]
    nxtState[47+1] = preds[1]
    nxtState[53+1] = preds[2]
    nxtState[59+1] = preds[3]
    nxtState[63+1] = preds[4]
    nxtState[64+1] = preds[5]
    nxtState[60+1] = preds[6]
    nxtState[54+1] = preds[7]
    nxtState[48+1] = preds[8]
    nxtState[40+1] = preds[9]
    
    for i in range(0,(x.shape[0]-2)):
        x[i][0] = x[i+1][0]
    x[x.shape[0]-1][0] = nxtState
       
print("Done!")
# END