import numpy as np
#import matplotlib.pyplot as plt

#from IPython.core.display import HTML, display
#from IPython.display import display, Audio

import sys
import base64
import struct
import os
import fnmatch
import re
import socket

import scipy.io.wavfile as wav
import python_speech_features as p

import pandas as pd

import itertools
import editdistance

#####################################################

# imports
import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape, Lambda, Input
# from keras.layers.recurrent import _time_distributed_dense
from keras.optimizers import SGD, adam
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.data_utils import Sequence
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add, concatenate
import keras.callbacks
from keras.models import model_from_json

print(keras.__version__) ##be careful with 2.0.6 as 2.0.4 tested with CoreML

from sklearn.utils import shuffle


sys.path.append('./utils/')
from utils import clean, read_text, text_to_int_sequence, int_to_text_sequence
from char_map import char_map, index_map


#######################################################

''' file assumes you already have a folder called test_mfccs and 
    a trained model you can load

'''

# let's load in some data from the test_mfcc folders
x0 = np.loadtxt('./test_mfccs/mfcc_test_0.csv', dtype=float, delimiter=',')

print(x0)

###############################

# load json and create model
json_file = open('TRIMMED_ds_ctc_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into loaded model
loaded_model.load_weights("TRIMMED_ds_ctc_model_weights.h5")
print("Loaded TRIMMED model/weights from disk")
loaded_model.summary()


# K.set_learning_phase(0)
X = np.zeros([1, 778, 26])
# input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)
# # test_func = K.function([input_data, K.learning_phase()],[y_pred])

##### we are ready to fire the mfcc's into the net

### FIRE

# np.reshape(x0, (1,778,26))

y = np.expand_dims(x0, axis=0)
print y.shape
prediction = loaded_model.predict(y)

print(prediction)
