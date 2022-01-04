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
start_time = time.time()
#model = tf.keras.models.load_model('models\seg76')
model = tf.keras.models.load_model('models\GameA_acc43')
print("--- %s seconds ---" % (time.time() - start_time)) #debug.
print("Models loaded...")

# Get start input data for model prediction:
rangeI = np.arange(0,96)
dataset_x = pd.read_csv('input.txt', usecols=rangeI, engine='python')
dataset_x = dataset_x.values
dataset_x = dataset_x.astype('float32')
x = np.reshape(dataset_x, (dataset_x.shape[0],1,dataset_x.shape[1]))

start_time = time.time()
res = model.predict(x)
#for i in range(96):
#    res = model.predict(x)
print("--- %s seconds ---" % (time.time() - start_time)) #debug.

      
print("Done!")
# END