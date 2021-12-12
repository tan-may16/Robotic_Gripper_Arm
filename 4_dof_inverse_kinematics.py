import math as m
import numpy as np
import time
# import serial
import sys
#arduino= serial.Serial('COM5',9600)
time.sleep(2)
print('Connection to Arduino...')

#d1=float(input('Enter the length of link 1'))
#d2=float(input('Enter the length of link 2'))
#a3=float(input('Enter the length of link 3'))
#a4=float(input('Enter the length of link 4'))
#d5=float(input('Enter the length of link 5'))

L1=38.55
L2=140
L3=134
L4=90


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

theta234=m.atan(nz/az)
theta1=m.atan(py/px)

A=px-L4*m.cos(theta1)*m.cos(theta234)
B=py-L4*m.sin(theta1)*m.cos(theta234)
C=pz-L1-L4*m.sin(theta234)

theta3= (A**2+B**2+C**2-L2**2-L3**2)/(2*L2*L3)
a=L2+L3*m.cos(theta3)
b=L3*m.sin(theta3)

theta2=m.acos(C/((a**2)+(b**2))**0.5)+m.atan(a/b)
theta4=theta234-theta2-theta3

joint_angles=np.array([theta1,theta2,theta3,theta4])
joint_angles=180*m.pi*joint_angles
for i in range(len(joint_angles)-1):
    joint_angles[i]=joint_angles[i]%360

print(joint_angles)
