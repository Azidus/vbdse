# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:43:20 2021

@author: P-kid
"""
#!/usr/bin/python

# Standard imports
import cv2
import numpy as np

file = 'STATE_INIT.jpg'
#file = '\STATE_TEST1.png'
#file = '\square4.png'

findBlobs = False
findContours = True
numberContours = True
verbose = True

# Read image
im = cv2.imread(file)


# Invert image colors:
im = (255-im)

if findBlobs:
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 240
    params.maxThreshold = 255
    
    # 
    params.minDistBetweenBlobs = 10
    
    # Filter by Color.
    params.filterByColor = False;
    blobColor = 0;
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    
    # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = False
    #params.minInertiaRatio = 0.01
    
    
    # Create a detector with the parameters
    # OLD: detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)
    
    
    # Detect blobs.
    keypoints = detector.detect(im)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)

if findContours:
    #convert img to grey
    img_grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #create an empty image for contours
    img_contours = np.zeros(im.shape)
    # draw the contours on the empty image
    if verbose:
        for i in range(len(contours)):
        #for i in range(1,90):
            #print(contours[i])
            print(i)
            cv2.drawContours(img_contours, contours[i], -1, (255,255,255), -1)
            if numberContours:
                M = cv2.moments(contours[i])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img_contours, str(i), (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.fillPoly(img_contours, pts =contours[i], color=(255,255,255))
            cv2.imshow("Contours", img_contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
#contours = contours[4:5]
#cv2.drawContours(img_contours, contours, -1, (255,255,255), -1)
cv2.fillPoly(img_contours, pts =contours, color=(255,255,255))
cv2.imshow("Contours", img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("done!")

#END