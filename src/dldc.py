#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import os
import time
import rospy
import sys
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float64MultiArray

# Comment to use tensorflow
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

print("------------INITIALIZE DEPENDENCIES--------------")

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
import random
import scipy.io as sio
import glob
import ast
np.random.seed(1337) # for reproducibility

import keras
import keras.models as models
from keras.layers import LSTM, TimeDistributed
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop

from keras import backend as K
K.set_image_data_format("channels_first")

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from skimage import color, exposure, transform

def preprocess_img(img):
    img = img/np.max(np.max(img))
    return(img)

class Dldc():
    #Class attribute

    ros_path = '/home/introlab/ros_ws/src/dldc/src/'
    weights_path = '/home/introlab/ros_ws/src/dldc/src/weight/train11/'
    path = '/home/introlab/ros_ws/src/dldc/src/dataset/'
    img_channels = 1
    img_rows = 513
    img_cols = 20
    epochs = 1000
    batch_size = 32
    search = os.path.join(path , "train", "*", "*.mat" )
    files = len(glob.glob(search))
    searchval = os.path.join(path , "val", "*", "*.mat" )
    filesval = len(glob.glob(searchval))
    steps_per_epoch = np.floor(files/batch_size)
    steps_per_epoch_val = np.floor(filesval/batch_size)
    nb_class = 2
    nb_dim = 1
    frame = []
    start = 0

    #Model save variables
    save_model_name= weights_path + 'test9.hdf5'
    run_model_name= weights_path + 'test9.hdf5'
    load_model_name= weights_path + 'test9.hdf5'

    network = models.Sequential()
    bridge = CvBridge()

    def __init__(self):
        x=1
    #Data generator
    def prep_train_data(self):

        while 1:
            search = os.path.join(self.path , "train", "*", "*.mat" )
            files = glob.glob(search)
            files.sort()

            train_data = []
            train_label = []

            for i in range(self.batch_size):



                index= random.randint(0, len(files)-1)
                t = files[index].split('/')

                if t[-2] == "drone":
                    label = 1
                else:
                    label = 0

                train_label.append(label)
                data = sio.loadmat(files[index])
                im = data['im']
                data = np.rollaxis(preprocess_img(im),1)
                train_data.append(data)

            yield(np.array(train_data), np.array(train_label))

    def prep_val_data(self):

        while 1:
            search_val = os.path.join(self.path , "val", "*", "*.mat" )
            files_val = glob.glob(search_val)
            files_val.sort()

            val_data = []
            val_label = []

            for i in range(self.batch_size):

                index= random.randint(0, len(files_val)-1)

                t = files_val[index].split('/')

                if t[-2] == "drone":
                    label = 1
                else:
                    label = 0

                val_label.append(label)
                data = sio.loadmat(files_val[index])
                im = data['im']
                data = np.rollaxis(preprocess_img(im),1)
                val_data.append(data)

            yield(np.array(val_data), np.array(val_label))


    def create_network(self):
        #Model creation
        print("------------CREATING NETWORK--------------")

        self.network.add(LSTM(500, go_backwards=True, return_sequences=True, stateful=False, input_shape=(self.img_cols,self.img_rows)))
        self.network.add(LSTM(500, go_backwards=True, stateful=False))
        self.network.add(Dense(1))
        self.network.add(Activation('sigmoid'))

        optimizer = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.network.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
        self.network.summary()

    def train_network(self):
        print("------------TRAINING NETWORK--------------", self.save_model_name)

        #self.network.load_weights(self.load_model_name)

        logcb = keras.callbacks.ModelCheckpoint(\
        "/home/introlab/ros_ws/src/dldc/src/weight/train11/weights.{epoch:02d}-{val_acc:.2f}.hdf5", \
            monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, \
            mode='auto', period=1)

        #escb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, \
        #    patience=3, verbose=1, mode='auto')

        history = self.network.fit_generator(self.prep_train_data(), \
        epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, \
        validation_data=self.prep_val_data(), validation_steps=self.steps_per_epoch_val, \
        verbose=1, callbacks=[logcb])

        self.network.save_weights(self.save_model_name)
        print(history.history.keys())

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def deploy_network(self):

        print("------------DEPLOYING NETWORK--------------")

        #Deployment variables
        d_path = self.run_model_name
        self.network.load_weights(d_path)

    def image_analysis(self):
        #Image analysis
        import os

        data = sio.loadmat('/home/introlab/ros_ws/src/dldc/src/dataset/val/notdrone/' + 'not_drone0N_0.001.mat')
        img = data['im']
        img =  np.rollaxis(preprocess_img(img),1)
        img = img[np.newaxis,:]
        pred = self.network.predict(img, batch_size=1)
        print(pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def process_image(self, image):
        vid_img_prep = []
        vid_img = cv2.resize(image, (self.img_rows,self.img_cols))
        vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
        vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
        output = self.network.predict_proba(np.array(vid_img_prep)[1:2])
        pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows,self.img_cols)))
        return pred

    def invert_red_blue(image):
        return image[:,:,[2,1,0]]

    def video_analysis(self):
        #Video playback analysis
        video = VideoFileClip("01TP_extract.avi")
        video = video.fl_image(invert_red_blue)
        pred_video = video.fl_image(process_image)
        pred_video.write_videofile('pred_video.avi', codec='rawvideo', audio=False)

    def live_analysis(self):
        #Live stream video analysis
        while not rospy.is_shutdown():
            img = np.rollaxis(preprocess_img(self.frame),1)
            img = img[np.newaxis,:]
            pred = self.network.predict(img, batch_size=1)

            if pred > 0.5:
                print(pred, "DRONE DETECTION!")
            else:
                print(pred, ".")

    def str_callback(self, msg):
        data = sio.loadmat("stft_img.mat")
        frame = data['im']
        img = np.rollaxis(preprocess_img(frame),1)
        img = img[np.newaxis,:]
        pred = self.network.predict(img, batch_size=1)

        if pred > 0.5:
            print(pred, "DRONE DETECTION!")
        else:
            print(pred, ".")



if __name__ == '__main__':
    arg = sys.argv

    sn = Dldc()
    rospy.init_node('dldc_node_run', anonymous=True)

    sn.create_network()
    sn.train_network()
    #sn.deploy_network()
    #sn.image_analysis()
    #sn.live_analysis()


    rospy.Subscriber("/stft_img_flag", String, sn.str_callback)
    #rospy.Subscriber("/stereo_camera/left/image_rect_color", Image, sn.image_callback)


    try:
        rospy.spin()
    except KeyboardInterrupt:image_analysis
    print("Shutting down artificial neural network")
