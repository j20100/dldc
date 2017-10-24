#!/usr/bin/env python
from rtabmap_ros.msg import UserData
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64
from biobot_ros_msgs.msg import FloatList
import std_msgs.msg
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

import scipy.io as sio

bridge = CvBridge()

# Create STFT
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import os,glob
import scipy.misc
import sounddevice as sd
import json



if __name__ == '__main__':

    rospy.init_node('sound_img', anonymous=True)
    STFT_pub = rospy.Publisher("stft_img_flag", String, queue_size=10)
    fs = 44100
    np.set_printoptions(threshold=np.nan)
    bridge = CvBridge()
    sd.default.samplerate = fs
    sd.default.channels = 1
    sd.default.device = 7

    while not rospy.is_shutdown():
        #1024*10-512
        x = sd.rec(1024*10-512, blocking=True, dtype='float64')
        x =x.T[0,:]
        f, t, Zxx = signal.stft(x.T, fs, nperseg=1024)
        stft_img = np.log(np.abs(Zxx))
        sio.savemat("stft_img.mat", {"im": stft_img})
        STFT_pub.publish("Stft Img ready")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
