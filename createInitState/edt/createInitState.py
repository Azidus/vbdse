# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:43:20 2021

@author: Patrick
"""
#!/usr/bin/python

# IMPORT LIBRARIES:
import cv2
#import numpy as np

# SETUP:
thresh = 100
  
# MAIN PROGRAM:  
img1 = cv2.imread("1.jpg") # Read input image.
ret,thresh_img1 = cv2.threshold(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), thresh, 255, cv2.THRESH_BINARY)
img2 = cv2.imread("2.jpg") # Read input image.
ret,thresh_img2 = cv2.threshold(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), thresh, 255, cv2.THRESH_BINARY)
img3 = cv2.imread("3.jpg") # Read input image.
ret,thresh_img3 = cv2.threshold(cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY), thresh, 255, cv2.THRESH_BINARY)
img4 = cv2.imread("4.jpg") # Read input image.
ret,thresh_img4 = cv2.threshold(cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY), thresh, 255, cv2.THRESH_BINARY)
res = cv2.bitwise_and(thresh_img1,thresh_img2)
res = cv2.bitwise_and(res,thresh_img3)
res = cv2.bitwise_and(res,thresh_img4)
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("STATE_INIT.jpg", res)
print("done!")

#END