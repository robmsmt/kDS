import numpy as np

import sys
import base64
import struct
import os
import fnmatch
import re

import scipy.io.wavfile as wav
import python_speech_features as p

datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/timit/"
#datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LDC/timit/"
target = datapath+"TIMIT/"



# WE CONVERT THE WAV (NIST sphere format) into MSOFT WAV

# Note on Windows and Mac had caseconflict here which dropbox was enforcing so I added _rif.wav to make it work for all
# https://askubuntu.com/questions/46658/has-ubuntu-gone-case-insensitive

import subprocess

for root, dirnames, filenames in os.walk(target):
    for filename in fnmatch.filter(filenames, "*.WAV"):
        sph_file = os.path.join(root, filename)
        wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"

        print("converting {} to {}".format(sph_file, wav_file))
        subprocess.check_call(["sox", sph_file, wav_file])


