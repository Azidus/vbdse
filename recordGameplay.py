# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:12:22 2021

@author: Patrick
"""

# IMPORT NECCESSARY PACKAGES:
import numpy as np
import cv2
import win32gui # Only works on windows !!!
from mss import mss
from pynput import keyboard
#import uuid
import time
import os
import sys

"""
TODO::
    Stable framerate - millis timer.
"""

# GLOBAl VARIABLES:
bounding_box = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
windowName = "MAME: Game & Watch: Ball [gnw_ball]"
sct = mss()
#filepath = r'C:\Users\P-kid\OneDrive\Dokumenter\Work\DTU\Kandidat\mscSEM3\02456 - Deep learning F21\02456 - E21 project\dataset\i'
filepath = "dataset"
inputStr = "000"
inputLatch = 0
stopFlag = 0
loopTime_seconds = 0.1 
idx = 1000000000000000

# FUNCTIONS DEFINITIONS:
def on_press(key):
    global inputStr
    global inputLatch
    #try:
    #print(key)
    #print('alphanumeric key {0} pressed'.format(key.char))
    if inputLatch == 0:
        inputStr=""
        if key == keyboard.Key.right:
            inputStr+="1"
        else:
            inputStr+="0"
        if key == keyboard.Key.left:
            inputStr+="1"
        else:
            inputStr+="0"
        if(str(key)=="'1'"):
            #print("STart btn")
            inputStr+="1"
        else:
            inputStr+="0"
        inputLatch=1
    #except AttributeError:
        #print('special key {0} pressed'.format(key))

def on_release(key):
    global stopFlag
    #print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        stopFlag = 1
        # Stop listener
        return False    
    
def callback(hwnd, extra):
    global bounding_box
    if windowName in win32gui.GetWindowText(hwnd):
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        #print("Window %s:" % win32gui.GetWindowText(hwnd)) #debug.
        #print("\tLocation: (%d, %d)" % (x, y)) #debug.
        #print("\t    Size: (%d, %d)" % (w, h)) #debug.
        bounding_box = {'top': y, 'left': x, 'width': w, 'height': h}
        bounding_box = {'top': 75, 'left': 233, 'width': 1464, 'height': 920}

# SETUP:
win32gui.EnumWindows(callback, None) #TODO::crop window borders and color black BG.
if(os.path.isdir(filepath)==False or bounding_box == {'top': 0, 'left': 0, 'width': 0, 'height': 0}):
    sys.exit("ERROR1 - Startup error, terminating script !")
    
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()
    
# MAIN LOOP:
try:
    while stopFlag == 0:
        startTime = time.time()
        image = np.array(sct.grab(bounding_box))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        #cv2.imwrite(filepath+str(timeStamp)+"("+inputStr+").jpg", image)
        if(inputLatch==1): #If button input received:
            #image = cv2.putText(image, inputStr, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            #print(str(idx) + ", " + inputStr)
            cv2.imwrite(filepath+"\\"+str(idx)+"-"+inputStr+".jpg", image)
            inputLatch=0
        else: # else - if no input is registered:
            cv2.imwrite(filepath+"\\"+str(idx)+"-000"+".jpg", image)
        #inputStr = "000"
        idx+=1
        # Wait so loop only updates 5 times a second:
        while True:
            if(time.time()-startTime>loopTime_seconds):
                break
        
except KeyboardInterrupt:
        print("n\Exit")
print("Done!")
# END
