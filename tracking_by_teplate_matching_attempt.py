import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math as m
import time
import serial
import sys
#from pyfirmata import Arduino, SERVO
from time import sleep

#board = Arduino('COM6')#usb port
#board.digital[10].mode =SERVO
#board.digital[10].write(40)
#board.digital[3].mode =SERVO
#board.digital[3].write(15)
#board.digital[10].mode =SERVO
#board.digital[10].write(0)
#board.digital[3].mode =SERVO
#board.digital[3].write(50)
#board.digital[5].mode =SERVO
#board.digital[5].write(50)

sleep(2)

#arduino= serial.Serial('COM5',9600)
#time.sleep(2)
print('Connection to Arduino...')
template=cv2.imread('template cross.JPG',0)
w,h=template.shape[::-1]

cap=cv2.VideoCapture(0)
Pxold=0
Pyold=0
while(cap.isOpened()):
     x, frame=cap.read()
     if x==True:
         gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

         Res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
         limit = 0.55
         loc = np.where(Res >= limit)
         #print(loc)
         PxTotal=0
         PyTotal=0
         count=0

         for i in zip(*loc[::-1]):
             cv2.rectangle(frame, i, (i[0] + w, i[1] + h), (0, 0, 255), 1)
             PxCentre=i[0]+w/2
             PyCentre=i[1] + h/2
             PxTotal=PxTotal+PxCentre
             PyTotal=PyTotal+ PyCentre
             count=count+1
             Px=int(PxTotal/count)
             Py=int(PyTotal/count)
         print(Px,'',Py)
         cv2.circle(frame,(int(Px),int(Py)),1,(0,255,255),2)

         if Px>60 and Px < 360:
             if abs(Px-Pxold) >= 10:
                 board.digital[11].write((140+Px)/5)
                 sleep(0.01)

         if Py>60 and Py < 420:
             if abs(Py-Pyold) >= 4:
                 board.digital[3].write((Py-60)/2)
                 sleep(0.01)
         Pxold=Px
         Pyold=Py

         cv2.imshow("Video Window",frame)
         if cv2.waitKey(1)==ord('q'):
             break
     else:
         break

cap.release()
cv2.destroyAllWindows()





