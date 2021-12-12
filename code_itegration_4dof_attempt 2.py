import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math as m
import time
import serial
import sys
from pyfirmata import Arduino, SERVO
from time import sleep

print('Connection to Arduino...')

board = Arduino('COM6')#usb port
board.digital[11].mode =SERVO
board.digital[11].write(170)
board.digital[6].mode =SERVO
board.digital[6].write(0)
#board.digital[10].mode =SERVO
#board.digital[10].write(0)
board.digital[3].mode =SERVO
board.digital[3].write(50)
board.digital[5].mode =SERVO
board.digital[5].write(50)

sleep(2)



def rotate(pin, angle):

    board.digital[pin].write(angle)
    #sleep(2)

#L1=385.5
#L2=1400
#L3=1340
#L4=900

L1=192
L2=700
L3=670
L4=450


alphaD=float(input('Enter the angle of rotation about x'))
betaD=float(input('Enter the angle of rotation about y'))
gammaD=float(input('Enter the angle of rotation about z'))

px=float(input('Enter the Z coordinate of desired end effector Position:'))

alpha=alphaD*m.pi/180
beta=betaD*m.pi/180
gamma=gammaD*m.pi/180

cap=cv2.VideoCapture('cross test.mp4')
ret,frame=cap.read()
frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#plt.imshow(frame1)
#plt.show()

x,y,w,h=1040,370,130,130
track_window=(x,y,w,h)
roi=frame[y:y+h,x:x+w]
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv_roi,np.array((14.,0.,0.)),np.array((190.,255.,255.)))
roi_hist=cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
arr=[]

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 255], 1)
re, track_window = cv2.meanShift(dst, track_window, term_criteria)

x, y, w, h = track_window
final_image = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)
arr.append(track_window)

(pz, py) = (x + w / 2, y + h / 2)
print(pz, 'and', py)
#plt.imshow(frame1)
#plt.show()
joint_angles_old=[0,0,0,0]
while(cap.isOpened()):
     x, frame=cap.read()
     if x==True:
         hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

         dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
         ret,track_window=cv2.meanShift(dst,track_window,term_criteria)

         x,y,w,h=track_window
         final_image= cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
         arr.append(track_window)
         (delPz, delPy) = (pz - x - w / 2, py - y - h / 2)
         (pz,py)=(x+w/2,y+h/2)
         #print(delPz,'',delPy)
         #print(track_window)

         Final_matrix_1R = np.array(
             [m.cos(alpha) * m.cos(beta), (m.sin(beta) * m.sin(gamma) * m.cos(alpha)) - (m.sin(alpha) * m.cos(gamma)),
              m.sin(beta) * m.cos(gamma) * m.cos(alpha) + m.sin(alpha) * m.sin(gamma), px])
         Final_matrix_2R = np.array(
             [m.sin(alpha) * m.cos(beta), (m.sin(beta) * m.sin(gamma) * m.sin(alpha)) + (m.cos(alpha) * m.cos(gamma)),
              m.sin(beta) * m.cos(gamma) * m.sin(alpha) - m.cos(alpha) * m.sin(gamma), py])
         Final_matrix_3R = np.array([-m.sin(beta), m.cos(beta) * m.sin(gamma), m.cos(beta) * m.cos(gamma), pz])
         Final_matrix_4R = np.array([0, 0, 0, 1])
         Final_matrix = np.array([Final_matrix_1R, Final_matrix_2R, Final_matrix_3R, Final_matrix_4R])
         # print(Final_matrix)
         nx = Final_matrix[0][0]
         ny = Final_matrix[1][0]
         nz = Final_matrix[2][0]
         ox = Final_matrix[0][1]
         oy = Final_matrix[1][1]
         oz = Final_matrix[0][1]
         ax = Final_matrix[0][2]
         ay = Final_matrix[1][2]
         az = Final_matrix[2][2]

         theta234 = m.atan(nz / az)
         theta1 = m.atan(py / px)

         A = px - L4 * m.cos(theta1) * m.cos(theta234)
         B = py - L4 * m.sin(theta1) * m.cos(theta234)
         C = pz - L1 - L4 * m.sin(theta234)

         theta3 = (A ** 2 + B ** 2 + C ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
         a = L2 + L3 * m.cos(theta3)
         b = L3 * m.sin(theta3)

         theta2 = m.acos(C / ((a ** 2) + (b ** 2)) ** 0.5) + m.atan(a / b)
         theta4 = theta234 - theta2 - theta3


         joint_angles = np.array([theta1, theta2, theta3, theta4])

         for i in range(len(joint_angles)):
             joint_angles[i] = int(m.ceil((180 * joint_angles[i]/ m.pi )%360))
             if joint_angles[1]>40 and joint_angles[1]<100:
                 if abs(joint_angles_old[1]-joint_angles[1])>=2:
                     board.digital[3].write(joint_angles[1])
                     sleep(0.01)
             if joint_angles[2]>15 and joint_angles[2]<180:
                 if abs(joint_angles_old[2] - joint_angles[2]) >= 2:
                     board.digital[11].write(joint_angles[2])
                     sleep(0.01)
             #if joint_angles[3]>15 and joint_angles[3]<180:
              #   board.digital[10].write(joint_angles[2])
               #  sleep(0.01)
             if joint_angles[0]>0 and joint_angles[0]<180:
                 if abs(joint_angles_old[2] - joint_angles[2]) >= 2:
                     board.digital[3].write(joint_angles[2])
                     sleep(0.01)

         print(joint_angles)
         joint_angles_old=joint_angles
         cv2.imshow("Video Window",final_image)
         k=cv2.waitKey(10)
         if k==27:
             break
     else:
         break

DF = pd.DataFrame(arr)
DF.to_csv("data1.csv")
cap.release()
cv2.destroyAllWindows()

