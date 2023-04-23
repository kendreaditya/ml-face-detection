#import libraries

import cv2
import numpy as np 

#Create video capture object

vid = cv2.VideoCapture(0)
arr = []

cv2.namedWindow("Window")

while(vid.isOpened()):

    #capture video frame by frame
    ret, frame = vid.read()
    arr.append(frame)
    cv2.imshow('Window', frame)

    #quit the script using the q key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
    	print(len(arr)) 
    	break


#Release video capture object
vid.release()

#destroy all windows
cv2.destroyAllWindows()