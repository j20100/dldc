# Create STFT
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import os,glob
import scipy.misc
import random

np.random.seed(1337)

cwd = os.getcwd()
searchfilesdrone = os.path.join(cwd,'dataset','train','drone','*.mat')
searchfilesnotdrone = os.path.join(cwd,'dataset','train','notdrone','*.mat')

filesdrone = glob.glob(searchfilesdrone)
filesnotdrone = glob.glob(searchfilesnotdrone)

val_size_drone = floor(len(filesdrone)*0.15)
val_size_not_drone = floor(len(filesnotdrone)*0.15)

for i in range(val_size_drone):
    index= random.randint(0, len(filesdrone)-1)

    k = filesdrone[i].split('/')

    output_name = '/'.join(map(str,k[:-3])) + '/val/' + '/'.join(map(str,k[-2:]))

    os.rename(filesdrone[i], output_name)

for i in range(val_size_not_drone):
    index= random.randint(0, len(filesnotdrone)-1)

    k = filesnotdrone[i].split('/')

    output_name = '/'.join(map(str,k[:-3])) + '/val/' + '/'.join(map(str,k[-2:]))

    os.rename(filesnotdrone[i], output_name)
