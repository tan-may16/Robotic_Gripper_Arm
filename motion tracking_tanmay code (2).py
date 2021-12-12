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
# temp_frame = cv2.imread('temp_red2.jpg')
# xt,yt,wt,ht=627,302,140,140

temp_frame = cv2.imread('temp_tanmay.jpeg')
xt,yt,wt,ht=988,304,230,214
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

#
# fourcc= cv2.VideoWriter_fourcc(*'XVID')
# out= cv2.VideoWriter('output_camshift.avi', fourcc, 40.0, (640,480))

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

            #lower boundary RED color range values; Hue (0 - 10)
            lower1 = np.array([0, 100, 10])
            upper1 = np.array([10, 255, 255])

            # upper boundary RED color range values; Hue (160 - 180)
            lower2 = np.array([130,150,20])
            upper2 = np.array([180,255,255])

            lower_mask = cv2.inRange(hsv_roi, lower1, upper1)
            upper_mask = cv2.inRange(hsv_roi, lower2, upper2)

            full_mask = lower_mask + upper_mask

            #full_mask=cv2.inRange(hsv_roi,np.array((115.,150,10)),np.array((180.,255.,255)))
            #full_mask=cv2.inRange(hsv_roi,np.array((0.,0.,0.)),np.array((255.,255.,255.)))
            roi_hist=cv2.calcHist([hsv_roi],[0],full_mask,[256],[0,256])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            term_criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
            arr=[]




    elif check==True:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
        # ret,track_window=cv2.meanShift(dst,track_window,term_criteria)
        # x,y,w,h=track_window
        # frame= cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        ret, track_window = cv2.meanShift(dst, track_window, term_criteria)
    #    draw it on image
        pts = cv2.boxPoints(ret)
    #    print(pts)
        pts = np.int0(pts)

        frame = cv2.polylines(frame, [pts], True, (0,255,0), 2)
        cv2.imshow('dst', dst)

        cv2.imshow("Video Window",frame)

    else:
        cv2.imshow("Video Window",frame)

    # out.write(frame)

    k=cv2.waitKey(10)
    if k==27:
        break
    # else:
    #     break
#print(arr)
cap.release()
cv2.destroyAllWindows()
