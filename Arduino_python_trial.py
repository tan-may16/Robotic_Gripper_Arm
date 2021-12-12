import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math as m
import time
import serial
import sys

usbport = 'COM6'
arduino = serial.Serial(usbport, 345600)
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


move(3, 30)