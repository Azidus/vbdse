# -*- coding: utf-8 -*-

# IMPORT LIBRARIES:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "GameA_RNN1-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), write_grads = True)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# SETUP:   
# Prepare train data:
#range1 = [i for i in range(0,95)]
rangeI = np.arange(0,96)
rangeO = [1+1,2+1,3+1,4+1,29+1,30+1,33+1,36+1,37+1,38+1,39+1,40+1,41+1,45+1,46+1,47+1,48+1,49+1,50+1,51+1,52+1,53+1,54+1,55+1,56+1,57+1,58+1,59+1,60+1,61+1,62+1,63+1,64+1,68+1,69+1,80+1,81+1,88+1]#np.arange(0,93)#[76+1]
dataset_x = pd.read_csv('df.txt', usecols=rangeI, engine='python')
dataset_x = dataset_x.values
dataset_x = dataset_x.astype('float32')
x = np.reshape(dataset_x, (dataset_x.shape[0],1,dataset_x.shape[1]))
dataset_y = pd.read_csv('labels.txt', usecols=rangeO, engine='python')   
dataset_y = dataset_y.values
dataset_y = dataset_y.astype('float32')
#y = np.reshape(dataset_y, (dataset_y.shape[0],1,dataset_y.shape[1]))
y = np.reshape(dataset_y, (dataset_y.shape[0],dataset_y.shape[1]))
x_train = x
y_train = y 
# Prepare test data:
#x_test = pd.read_csv('test_ball.txt', usecols=rangeI, engine='python')
x_test = pd.read_csv('df_test.txt', usecols=rangeI, engine='python')
x_test = x_test.values
x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))
#y_test = pd.read_csv('test_ball.txt', usecols=[96], engine='python')
y_test = pd.read_csv('labels_test.txt', usecols=rangeO, engine='python')   
y_test = y_test.values
y_test = y_test.astype('float32')
y_test = np.reshape(y_test, (y_test.shape[0],y_test.shape[1]))
# Quick test:
#x_test = x_train
#y_test = y_train

num_in = x_train.shape[2]
num_out = y_train.shape[1]

# Build LSTM model:
model = keras.models.Sequential()
model.add(keras.Input(shape=(None, num_in))) #timesteps, seq_length.
#model.add(layers.LSTM(128, dropout=0.1))
model.add(layers.LSTM(128, dropout=0.1))
model.add(layers.Dense(num_out, activation="sigmoid"))
#print(model.summary)

model.compile(
    loss = keras.losses.BinaryCrossentropy(from_logits=False),
    #optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6),
    optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6),
    metrics = ["accuracy"]    
)

batchSize = 32
verbose = 2
# Train model:
print("Training model ...")
model.fit(x_train, y_train, batch_size=batchSize, epochs=30, verbose=verbose, callbacks=[tensorboard])

# Evaluate model:
print("Done training. Testing model ...")
model.evaluate(x_test, y_test, batch_size=batchSize, verbose=verbose)

# SAVE MODEL:
#model.save('models\tfmodel')

print("Done!")    
# END.
