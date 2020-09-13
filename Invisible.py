# Importing the required modules to do the invisible task

import cv2   # The module which will deals with the images
import numpy as np    # The module which will deal with image's pixel array
import time   # The module which will deal with the time

# We are going to use this time module to show the exact time on our videoscreen

# Now I am going to create a video capture oblect which will capture the videos from my webcam

cap = cv2.VideoCapture(0)  # I am going to use my integrated webcam that's why I have use 0 if you do not have you integrated webcam then you can use 1 as the parameter

# The video consists of the infinite frame so we have to make use of the while loop until we dont required the video input.

time.sleep(3)
# Now I am going to provide the camera 3 sec  to setup accrding the environment

background = 0

# Now we are going to give 30 iteration to the camera to capture the backgorund

for i in range(30):

    ret, background = cap.read()

# Now we are approching to capture ouself in the form of array

while cap.isOpened(): # This loop will end only when we will close the webcam 

    # .read() returns two value one is in boolean and other the frame of the capture
    ret, img = cap.read()

    # If ret is false i.e the read function does not work then in that case this loop will exit
    if not ret: 
        break

    # Now we are going to convert the img to hsv which is in BGR form
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Now we are defining the two array
    
    # I have used yellow color cloth we can choose your

    lower = np.array([20,100,100])
    upper = np.array([30,255,255])

    # This will create a mask in this range
    mask1 =  cv2.inRange(hsv,lower,upper)

    lower = np.array([170,100,100])
    upper = np.array([180,255,255])
    
    # Same as the mask1 
    mask2 =  cv2.inRange(hsv,lower,upper)

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 2)
    
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8), iterations = 1)

    mask2 = cv2.bitwise_not(mask1)
    
    res1 = cv2.bitwise_and(background,background,mask = mask1)
    res2 = cv2.bitwise_and(img,img,mask = mask2)

    final = cv2.addWeighted(res1,1,res2,1,0)

    cv2.imshow('Final',final)
    k = cv2.waitKey(10)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
    
