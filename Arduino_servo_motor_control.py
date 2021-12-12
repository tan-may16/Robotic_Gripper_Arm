import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math as m
import time
import serial
import sys



d1=550
d2=385.5
a3=1400
a4=1341
d5=900
alphaD=float(input('Enter the angle of rotation about x'))
betaD=float(input('Enter the angle of rotation about y'))
gammaD=float(input('Enter the angle of rotation about z'))
pz=float(input('Enter the Z coordinate of desired end effector Position:'))

alpha=alphaD*m.pi/180
beta=betaD*m.pi/180
gamma=gammaD*m.pi/180
usbport = 'COM6'
arduino = serial.Serial(usbport, 345600)
#time.sleep(2)
print('Connection to Arduino...')


def move(servo, angle):
    if (0 <= angle <= 180):
        arduino.write((255))
        arduino.write((servo))
        arduino.write((angle))
    else:
        print("Servo angle must be an integer between 0 and 180.\n")


def init():
    move(1, 90)
    move(2, 90)
    move(3, 90)
    move(4, 90)
    move(5, 90)


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

(px, py) = (x + w / 2, y + h / 2)
print(px, 'and', py)
#plt.imshow(frame1)
#plt.show()

while(cap.isOpened()):
     x, frame=cap.read()
     if x==True:
         hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

         dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,255],1)
         ret,track_window=cv2.meanShift(dst,track_window,term_criteria)

         x,y,w,h=track_window
         final_image= cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
         arr.append(track_window)
         (delPx, delPy) = (px - x - w / 2, py - y - h / 2)
         (px,py)=(x+w/2,y+h/2)
         print(delPx,'',delPy)
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
         theta12 = m.atan(-ox / oy)
         # print(theta12)
         theta2 = m.atan(py / px)
         # print(theta2)
         theta345 = m.atan((-az) / (m.cos(theta2) * ax + m.sin(theta2) * ay))
         # print(theta345)
         theta1 = theta12 - theta2
         # print(theta1)
         # costheta4=(((m.cos(theta12)*px)+(m.sin(theta12)*py))**2+(pz)**2 - (a3)**2+ ((a4)**2))/(2*a3*a4)
         costheta4 = ((m.cos(theta12) * px + m.sin(theta12) * py) ** 2 + (pz) ** 2 - (a3) ** 2 + (a4) ** 2) / (
                     2 * a3 * a4)
         # print(costheta4)
         sintheta4 = (1 - (costheta4) ** 2) ** 0.5
         # print(sintheta4)
         theta4 = m.atan(sintheta4 / costheta4)
         theta34 = -m.atan((m.cos(theta12) * px + m.sin(theta12) * py) / pz)
         theta3 = m.atan(((pz - a4 * m.sin(theta34)) / a3) / (
                     (m.cos(theta12) * px + m.sin(theta12) * py - a4 * m.cos(theta34)) / a3))
         theta5 = theta345 - theta34
         joint_angles = np.array([theta1, theta2, theta3, theta4, theta5])
         joint_angles = 180 * m.pi * joint_angles
         for i in range(len(joint_angles) - 1):
             joint_angles[i] = (joint_angles[i])%180
             move(i+1,joint_angles[i])


         print(joint_angles)
         #arduino.write(joint_angles)
         cv2.imshow("Video Window",final_image)
         k=cv2.waitKey(1000)
         if k==27:
             break
     else:
         break
#print(arr)


DF = pd.DataFrame(arr)
DF.to_csv("data1.csv")
cap.release()
cv2.destroyAllWindows()











