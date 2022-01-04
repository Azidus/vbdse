# -*- coding: utf-8 -*-

# IMPORT LIBRARIES:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from pynput import keyboard
import time
# IMPORT EXTERNAL PYTHON FILES:
import LCD_segment_detector

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# GLOBAL VARIABLES:
inputStr = [0,0,0]
inputLatch = 0
stopFlag = 0

# FUNCTIONS DEFINITIONS:
def on_press(key):
    global inputStr
    global inputLatch
    #try:
    #print(key)
    #print('alphanumeric key {0} pressed'.format(key.char))
    if inputLatch == 0:
        #inputStr=""
        inputStr=[]
        if key == keyboard.Key.right:
            #inputStr+="1"
            inputStr.append(1)
        else:
            #inputStr+="0"
            inputStr.append(0)
        if key == keyboard.Key.left:
            #inputStr+="1"
            inputStr.append(1)
        else:
            #inputStr+="0"
            inputStr.append(0)
        if(str(key)=="'1'"):
            #print("STart btn")
            #inputStr+="1"
            inputStr.append(1)
        else:
            #inputStr+="0"
            inputStr.append(0)
        inputLatch=1
    #except AttributeError:
        #print('special key {0} pressed'.format(key))

def on_release(key):
    global stopFlag
    #print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        stopFlag = 1
        print("Stopping listener...")
        # Stop listener
        return False 


# SETUP:
loopTime_seconds = 0.1 
idx = 0
# Get calibration image with all segments turned ON:
calib_path = 'STATE_INIT.jpg'
initState = cv2.imread(calib_path)
allSegments = LCD_segment_detector.getLCDSegmentStates(initState)

# Recreate the exact same model, including its weights and the optimizer
print("Loading models...")
start_time = time.time()
loaded_model = tf.keras.models.load_model('models\handtest1')
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
mseg73 = tf.keras.models.load_model('models\seg73')
megg3 = tf.keras.models.load_model('models\egg3')
megg4 = tf.keras.models.load_model('models\egg4')
print("--- %s seconds ---" % (time.time() - start_time)) #debug.
print("Models loaded - starting program...")
# Show the model architecture
#loaded_model.summary()

# Get start input data for model prediction:
#rangeI = np.arange(0,96)
#dataset_x = pd.read_csv('input.txt', usecols=rangeI, engine='python')
#dataset_x = dataset_x.values
#dataset_x = dataset_x.astype('float32')
#x = np.reshape(dataset_x, (dataset_x.shape[0],1,dataset_x.shape[1]))
#dataset_x = [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0]
dataset_x = [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0]

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

# MAIN LOOP:
try:
    while stopFlag == 0:
        startTime = time.time()
        # APPEND INPUT TO LIST.
        if inputLatch == 1:
            #print("btnpress:" + str(inputStr))
            #dataset_x.extend(inputStr)
            dataset_x[93] = inputStr[0]
            dataset_x[94] = inputStr[1]
            dataset_x[95] = inputStr[2]
            #inputStr = [0,0,0]
            inputLatch=0
        else:
            dataset_x[93] = 0
            dataset_x[94] = 0
            dataset_x[95] = 0

        # Generate predictions for samples
        #start_time = time.time() #debug.
        x = np.reshape(dataset_x, (1,1,len(dataset_x)))
        predictions = loaded_model.predict(x)
        #print("--- %s seconds ---" % (time.time() - start_time)) #debug.
        res = [int(round(predictions[0][0])),int(round(predictions[0][1])),int(round(predictions[0][2]))]
        #print(res) #debug.
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
        seg73 = mseg73.predict(x)
        egg3 = megg3.predict(x)
        egg4 = megg4.predict(x)

        # EDIT LIST BASED ON PREDICTION.
        dataset_x[33+1] = res[2]
        dataset_x[30+1] = res[1]
        dataset_x[29+1] = res[0]
        dataset_x[34+1] = dataset_x[29+1]
        dataset_x[32+1] = dataset_x[30+1]
        dataset_x[31+1] = dataset_x[33+1]
        # Legs:
        dataset_x[12+1] = dataset_x[31+1]
        dataset_x[13+1] = dataset_x[29+1]
        dataset_x[1+1] = abs(dataset_x[12+1]-1)
        dataset_x[2+1] = abs(dataset_x[13+1]-1)
        #stateBitstring = ''.join([str(elem) for elem in dataset_x]) # Convert list into string.
        # Ball:
        dataset_x[36+1] = int(round(ball36[0][0]))
        dataset_x[47+1] = int(round(ball47[0][0]))
        dataset_x[53+1] = int(round(ball53[0][0]))
        dataset_x[59+1] = int(round(ball59[0][0]))
        dataset_x[63+1] = int(round(ball63[0][0]))
        dataset_x[64+1] = int(round(ball64[0][0]))
        dataset_x[60+1] = int(round(ball60[0][0]))
        dataset_x[54+1] = int(round(ball54[0][0]))
        dataset_x[48+1] = int(round(ball48[0][0]))
        dataset_x[40+1] = int(round(ball40[0][0]))
        # Score:
        dataset_x[73+1] = int(round(seg73[0][0]))
        # Crush1:
        dataset_x[3+1] = int(round(egg3[0][0]))
        dataset_x[5+1] = dataset_x[3+1]
        dataset_x[6+1] = dataset_x[3+1]
        dataset_x[16+1] = dataset_x[3+1]
        dataset_x[19+1] = dataset_x[3+1]
        dataset_x[22+1] = dataset_x[3+1]
        dataset_x[23+1] = dataset_x[3+1]
        dataset_x[25+1] = dataset_x[3+1]
        dataset_x[26+1] = dataset_x[3+1]
        dataset_x[28+1] = dataset_x[3+1]
        dataset_x[35+1] = dataset_x[3+1]
        dataset_x[27+1] = dataset_x[3+1]
        # Crush2:
        dataset_x[4+1] = int(round(egg4[0][0]))
        dataset_x[9+1] = dataset_x[4+1]
        dataset_x[10+1] = dataset_x[4+1]
        dataset_x[17+1] = dataset_x[4+1]
        dataset_x[0+1] = dataset_x[4+1]
        dataset_x[7+1] = dataset_x[4+1]
        dataset_x[8+1] = dataset_x[4+1]
        dataset_x[11+1] = dataset_x[4+1]
        dataset_x[14+1] = dataset_x[4+1]
        dataset_x[18+1] = dataset_x[4+1]
        dataset_x[24+1] = dataset_x[4+1]
        dataset_x[15+1] = dataset_x[4+1]
        
        stateBitstring = dataset_x[:-3]
        # Convert bitstring representation of predicted state from RNN to contours to be drawn later: 
        nxtState = LCD_segment_detector.bitstring2Segments(allSegments, stateBitstring)

        # draw and show results:
        img_contours = np.zeros(initState.shape)            
        cv2.fillPoly(img_contours, pts=nxtState, color=(255,255,255))
        cv2.imshow("Game A", img_contours)
        cv2.waitKey(1)
        # Wait so loop only updates 5 times a second:
        while True:
            if(time.time()-startTime>loopTime_seconds):
                break
        #idx+=1
        #print(idx)
        
except KeyboardInterrupt:
        print("n\Exit")
        
cv2.destroyAllWindows()
print("Done!")
# END