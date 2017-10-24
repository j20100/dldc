# Create STFT
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import os,glob
import scipy.misc
import scipy.io as sio
import random


cwd = os.getcwd()

searchfilestodel = os.path.join(cwd,'dataset','*','*','*.mat')
del_those = glob.glob(searchfilestodel)

for i in del_those:
    os.remove(i)

searchfilesdrone = os.path.join(cwd,'dataset','train','drone','*.flac')
searchfilesnotdrone = os.path.join(cwd,'dataset','train','notdrone','*.flac')

filesdrone = glob.glob(searchfilesdrone)
filesnotdrone = glob.glob(searchfilesnotdrone)

allfiles = filesdrone + filesnotdrone

for file in allfiles:
    with open(file, 'rb') as f:
        x, fs = sf.read(f)

    k = file.split('/')
    output_path = k[:-1]

    # add white gaussian random noise
    Noises = [0, .001]
    for i in Noises:
        x += i*np.random.normal(0, 1, len(x))
        f, t, Zxx = signal.stft(x, fs, nperseg=1024)

        log_stft = np.log(np.abs(Zxx))

        im_col = 20
        hop = im_col/2

        for c in range(0, log_stft.shape[1], hop):
            im = log_stft[:, c:c + im_col]
            if im.shape[1] == im_col:
                output_name = '/'.join(map(str,k[:-1])) + '/' + k[-1][:-5] + str(c) + 'N_'+str(i)+'.mat'
                sio.savemat(output_name, {"im": im})



searchfilesdronemat = os.path.join(cwd,'dataset','train','drone','*.mat')
searchfilesnotdronemat = os.path.join(cwd,'dataset','train','notdrone','*.mat')

filesdronemat = glob.glob(searchfilesdronemat)
filesnotdronemat = glob.glob(searchfilesnotdronemat)

val_size_drone = int(len(filesdronemat)*0.15)
val_size_not_drone = int(len(filesnotdronemat)*0.15)
print(len(filesdronemat),len(filesnotdronemat))
print(val_size_drone,val_size_not_drone)

for i in range(val_size_drone):
    index= random.randint(0, len(filesdronemat)-1)

    k = filesdronemat[i].split('/')

    output_name = '/'.join(map(str,k[:-3])) + '/val/' + '/'.join(map(str,k[-2:]))

    os.rename(filesdronemat[i], output_name)

for i in range(val_size_not_drone):
    index= random.randint(0, len(filesnotdronemat)-1)

    k = filesnotdronemat[i].split('/')

    output_name = '/'.join(map(str,k[:-3])) + '/val/' + '/'.join(map(str,k[-2:]))

    os.rename(filesnotdronemat[i], output_name)
