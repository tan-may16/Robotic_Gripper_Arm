import cv2
import numpy as np
from matplotlib import pyplot as plt

cap=cv2.VideoCapture('cross test.mp4')
ret,frame=cap.read()
x,y,w,h=1040,370,130,130
track_window=[x,y,w,h]
roi=frame[y:y+h,x:x+w]
def nothing(x):
    pass
cv2.namedWindow('Trackbars')
cv2.createTrackbar("LH","Trackbars",0,255,nothing)
cv2.createTrackbar("LS","Trackbars",0,255,nothing)
cv2.createTrackbar("LV","Trackbars",0,255,nothing)
cv2.createTrackbar("HH","Trackbars",255,255,nothing)
cv2.createTrackbar("HS","Trackbars",255,255,nothing)
cv2.createTrackbar("HV","Trackbars",255,255,nothing)
while True:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    hh = cv2.getTrackbarPos("HH", "Trackbars")
    hs = cv2.getTrackbarPos("HS", "Trackbars")
    hv = cv2.getTrackbarPos("HV", "Trackbars")
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([hh, hs, hv])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #cv2.imshow('',roi)
    cv2.imshow('mask',mask)
    Final = cv2.bitwise_and(roi, roi, mask=mask)
    cv2.imshow('Final', Final)

    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.imwrite('Cross template.png', Final)
cap.release()
cv2.destroyAllWindows()
