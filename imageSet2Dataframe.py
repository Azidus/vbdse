# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:12:22 2021

@author: Patrick
"""

# IMPORT NECCESSARY PACKAGES:
import cv2
import os
# IMPORT EXTERNAL PYTHON FILES:
import LCD_segment_detector

# GLOBAL VARIABLES:
filetype = ".jpg"
directory = "dataset"
allSegments = LCD_segment_detector.getLCDSegmentStates(cv2.imread('STATE_INIT.jpg'))

# SETUP:
path, dirs, files = next(os.walk(directory))
file_count = len(files)
idx=1

f = open("dataframe.txt",'w')

# MAIN PROGRAM:
for filename in os.listdir(directory):
    if filename.endswith(filetype):
        #print(os.path.join(filename))
        img = cv2.imread(directory + "\\" + filename)
        currentState = LCD_segment_detector.getLCDSegmentStates(img)
        bitString = LCD_segment_detector.segments2Bitstring(allSegments, currentState)
        f.write(str(idx) + "," + bitString + "," + filename[-7:-4])
        f.write('\n')
        print("Converted file",idx, "out of",file_count)
        idx+=1
        continue
    else:
        continue    
f.close()

print("Done!")
# END
