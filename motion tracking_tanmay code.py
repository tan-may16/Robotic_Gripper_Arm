import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math as m
import time
import serial
import sys

cap=cv2.VideoCapture(0)
ret,frame=cap.read()
#frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#plt.imshow(frame1)
#plt.show()
temp_frame = cv2.imread('temp_red2.jpg')
xt,yt,wt,ht=627,302,140,140
#track_window_temp=(xt,yt,wt,ht)
temp=temp_frame[yt:yt+ht,xt:xt+wt]
cv2.imshow('temp',temp)
# hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# mask=cv2.inRange(hsv_roi,np.array((14.,0.,0.)),np.array((190.,255.,255.)))
# roi_hist=cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# term_criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
# arr=[]
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 255], 1)
# re, track_window = cv2.meanShift(dst, track_window, term_criteria)
# x, y, w, h = track_window
# final_image = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
# arr.append(track_window)
# (px, py) = (x + w / 2, y + h / 2)
# print(px, 'and', py)
# #plt.imshow(frame1)
#plt.show()
check= False

while(cap.isOpened()):
    x, frame=cap.read()
    # if x==True:
    res= cv2.matchTemplate(frame,temp, cv2.TM_CCOEFF_NORMED)
    th = 0.80



    if np.any(res>= th):

        loc = np.where(res>= th)


        for pt in zip(*loc[::-1]):
            frame= cv2.rectangle(frame, pt, (pt[0]+wt, pt[1]+ht), (0,0,255),2 )


        check = True
        cv2.imshow("Video Window",frame)
        loc_max = np.where(res.max())
        print(loc_max)
        for pt in zip(*loc[::-1]):
            x,y,w,h=pt[0],pt[1],130,130
            track_window=(x,y,w,h)
            roi=frame[y:y+h,x:x+w]
            hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            #mask=cv2.inRange(hsv_roi,np.array((101.,128.,0.)),np.array((148.,235.,124.)))
            mask=cv2.inRange(hsv_roi,np.array((0.,0.,0.)),np.array((255.,255.,255.)))
            roi_hist=cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            term_criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
            arr=[]




    elif check==True:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
        ret,track_window=cv2.meanShift(dst,track_window,term_criteria)
        x,y,w,h=track_window
        frame= cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow("Video Window",frame)

    else:
        cv2.imshow("Video Window",frame)

    k=cv2.waitKey(40)
    if k==27:
        break
    # else:
    #     break
#print(arr)
cap.release()
cv2.destroyAllWindows()
