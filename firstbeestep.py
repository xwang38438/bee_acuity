import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

import argparse
import time

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, type=str,
        help="file name for image")
ap.add_argument("-b", "--black", required=False, default=110, type=int,
        help="threshold below which is black")
ap.add_argument("-w", "--white", required=False, default=255, type=int,
        help="threshold above which is white")
ap.add_argument("-n", "--negative", required=False, default=1, type=int,
        help="-n 1 = make negative -n 0 not negative image")
ap.add_argument("-r1", "--r1", required=False, default=1, type=int,
        help="row start for cropping")
ap.add_argument("-r2", "--r2", required=False, default=500, type=int,
        help="row end for cropping")
ap.add_argument("-c1", "--c1", required=False, default=1, type=int,
        help="column start for cropping")
ap.add_argument("-c2", "--c2", required=False, default=500, type=int,
        help="column end for cropping")
args = vars(ap.parse_args())
file = args["file"]
black = args["black"]
white = args["white"]
negative = args["negative"]
r1 = args["r1"]
r2 = args["r2"]
c1 = args["c1"]
c2 = args["c2"]



# reading image
img = cv2.imread(file)
crop = img[r1:r2, c1:c2]

  
# converting cropped image into grayscale image
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
if negative == 1:
     gray = 255 - gray	#Can use negative

# setting threshold of cropped image
_, threshold = cv2.threshold(gray, black, white, cv2.THRESH_BINARY)
  
# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
i = 0


#cv2.imshow('cropped negative', gray)
#cv2.imshow('full', gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#sys.exit()

  
# list for storing names of shapes
for contour in contours:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
    #print(str(len(approx)))

#Adding contours to cropped image      
    # using drawContours() function
    cv2.drawContours(crop, [contour], 0, (0, 0, 255), 1)
  
    # finding center point of shape
    M = cv2.moments(contour)
    center_coordinates = (1,1) 	#initialize as tuple
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        center_coordinates = (x, y)
    if len(approx) > 8 :
        cv2.circle(crop, center_coordinates, 2, (0,255,0), 2)
        print(str(center_coordinates))
    print(str(len(approx)))
# displaying the image after drawing contours
cv2.imshow('shapes', crop)
cv2.imshow('full', img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

