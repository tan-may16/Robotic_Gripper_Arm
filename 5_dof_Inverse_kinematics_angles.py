import math as m
import numpy as np
import time
import serial
import sys
#arduino= serial.Serial('COM5',9600)
time.sleep(2)
print('Connection to Arduino...')

#d1=float(input('Enter the length of link 1'))
#d2=float(input('Enter the length of link 2'))
#a3=float(input('Enter the length of link 3'))
#a4=float(input('Enter the length of link 4'))
#d5=float(input('Enter the length of link 5'))

d1=55
d2=38.55
a3=140
a4=134.1
d5=90

alphaD=float(input('Enter the angle of rotation about x'))
betaD=float(input('Enter the angle of rotation about y'))
gammaD=float(input('Enter the angle of rotation about z'))

alpha=alphaD*m.pi/180
beta=betaD*m.pi/180
gamma=gammaD*m.pi/180

px=float(input('Enter the X coordinate of desired end effector Position:'))
py=float(input('Enter the Y coordinate of desired end effector Position:'))
pz=float(input('Enter the Z coordinate of desired end effector Position:'))
Final_matrix_1R=np.array([m.cos(alpha)*m.cos(beta),(m.sin(beta)*m.sin(gamma)*m.cos(alpha))-(m.sin(alpha)*m.cos(gamma)),m.sin(beta)*m.cos(gamma)*m.cos(alpha)+m.sin(alpha)*m.sin(gamma),px])
Final_matrix_2R=np.array([m.sin(alpha)*m.cos(beta),(m.sin(beta)*m.sin(gamma)*m.sin(alpha))+(m.cos(alpha)*m.cos(gamma)),m.sin(beta)*m.cos(gamma)*m.sin(alpha)-m.cos(alpha)*m.sin(gamma),py])
Final_matrix_3R=np.array([-m.sin(beta),m.cos(beta)*m.sin(gamma),m.cos(beta)*m.cos(gamma),pz])
Final_matrix_4R=np.array([0,0,0,1])
Final_matrix=np.array([Final_matrix_1R,Final_matrix_2R,Final_matrix_3R,Final_matrix_4R])
#print(Final_matrix)
nx=Final_matrix[0][0]
ny=Final_matrix[1][0]
nz=Final_matrix[2][0]
ox=Final_matrix[0][1]
oy=Final_matrix[1][1]
oz=Final_matrix[0][1]
ax=Final_matrix[0][2]
ay=Final_matrix[1][2]
az=Final_matrix[2][2]
theta12=m.atan(-ox/oy)
#print(theta12)
theta2=m.atan(py/px)
#print(theta2)
theta345=m.atan((-az)/(m.cos(theta2)*ax+m.sin(theta2)*ay))
#print(theta345)
theta1=theta12-theta2
#print(theta1)
#costheta4=(((m.cos(theta12)*px)+(m.sin(theta12)*py))**2+(pz)**2 - (a3)**2+ ((a4)**2))/(2*a3*a4)
costheta4=((m.cos(theta12)*px+m.sin(theta12)*py)**2+(pz)**2 - (a3)**2+(a4)**2)/(2*a3*a4)
#print(costheta4)
sintheta4=(1-(costheta4)**2)**0.5
#print(sintheta4)
theta4=m.atan(sintheta4/costheta4)
theta34=-m.atan((m.cos(theta12)*px+m.sin(theta12)*py)/pz)
theta3=m.atan(((pz - a4*m.sin(theta34))/a3)/((m.cos(theta12)*px + m.sin(theta12)*py - a4*m.cos(theta34))/a3))
theta5=theta345-theta34
joint_angles=np.array([theta1,theta2,theta3,theta4,theta5])
joint_angles=180*m.pi*joint_angles
for i in range(len(joint_angles)-1):
    x=(joint_angles[i])//360
    joint_angles[i]=joint_angles[i]-x*360

print(joint_angles)
#print(theta2)
