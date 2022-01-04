# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:43:20 2021

@author: Patrick
"""
#!/usr/bin/python

# IMPORT LIBRARIES:
import cv2

# FUNCTION DEFINITIONS:
def getLCDSegmentStates(image):
    """ Finds every contour in input image as indication for which LCD segments are turned ON. """
    # Convert image to greyscale:
    img_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Invert image colors:
    image = (255-image)
    # Set a threshold:
    thresh = 100
    # Get threshold image:
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # Find contours:
    #contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def segments2Bitstring(initState, curretnState):
    """ Compares initState(all segments ON), with current state and returns a bitstring representing which LCD segments are turned ON. """
    #bitString = []
    bitString = ""
    bit = 0
    nSegments = len(initState)
    for i in range (nSegments):
        for n in range (len(curretnState)):
            #if str(initState[i]) == str(curretnState[n]):
            if initState[i][0][0][0] == curretnState[n][0][0][0] and initState[i][0][0][1] == curretnState[n][0][0][1]:
                #bit = 1
                bit = '1'
                #initState.pop(i)
                #nSegments = len(initState)
                break
            else:
               #bit = 0
               bit = '0'
        #bitString.append(bit)
        bitString+=bit
    return bitString

def bitstring2Segments(initState, bitString):
    """ Takes in bitstring of predicted state and returns a list of contours based on which LCD segments are predicted to be ON. """
    contours = []
    for i in range(len(initState)):
        if bitString[i] == 1 or bitString[i] == '1':
        #if bitString[i] == '1':
            contours.append(initState[i])
    return contours
