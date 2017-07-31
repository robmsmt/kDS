from __future__ import unicode_literals

import numpy as np
import sys
import base64
import struct
import os
import fnmatch
import re

import pandas as pd


# if sys.version_info[0] == 3:
#     raise("Requires python 2.7")
# else:
#     pass


#####################################################

import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape, Lambda, Input
from keras.optimizers import SGD, adam
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.data_utils import Sequence

from keras.layers.merge import add, concatenate
import keras.callbacks

print(keras.__version__) ##be careful with 2.0.6 as 2.0.4 tested with CoreML

##### TEMP

from keras.models import model_from_json

# load json and create model
json_file = open('ds_ctc_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into loaded model
loaded_model.load_weights("ds_ctc_model_weights.h5")



##########################################

import coremltools
from keras.models import model_from_json

# load json and create model
json_file = open('ds_ctc_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into loaded model
loaded_model.load_weights("ds_ctc_model_weights.h5")
print("Loaded model/weights from disk")

# it looks like this has worked. We can now convert to
print("Convert Model")
try:
    coreml_model = coremltools.converters.keras.convert(loaded_model)
except Exception as e:
    print(e)


global batch_size
batch_size = 1

# Network Params
fc_size = 2048
rnn_size = 512
mfcc_features = 26
max_mfcclength_audio = 778

X = np.zeros([batch_size, max_mfcclength_audio, mfcc_features])

# Creates a tensor there are always 26 MFCC
input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)

# First 3 FC layers
x = Dense(fc_size, name='fc1', activation='relu',
          weights=loaded_model.layers[1].get_weights())(input_data) # >>(?, 778, 2048)
x = Dense(fc_size, name='fc2', activation='relu',
         weights=loaded_model.layers[2].get_weights())(x) # >>(?, 778, 2048)
x = Dense(fc_size, name='fc3', activation='relu',
         weights=loaded_model.layers[3].get_weights())(x) # >>(?, 778, 2048)

# Layer 4 BiDirectional RNN

rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', name='rnn_f',
                   weights=loaded_model.layers[4].get_weights())(x) #>>(?, ?, 512)

rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
                   kernel_initializer='he_normal', name='rnn_b',
                   weights=loaded_model.layers[5].get_weights())(x) #>>(?, ?, 512)

rnn_merged = add([rnn_1f, rnn_1b],
                weights=loaded_model.layers[6].get_weights()) #>>(?, ?, 512)
x = Activation('relu', name='birelu',
              weights=loaded_model.layers[7].get_weights())(rnn_merged) #>>(?, ?, 512)

# Layer 5 FC Layer
y_pred = Dense(fc_size, name='fc5', activation='relu',
              weights=loaded_model.layers[8].get_weights())(x) #>>(?, 778, 2048)

##############################################


#####################################

# Change shape
labels = Input(name='the_labels', shape=[80], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
# loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
#                                                                    labels,
#                                                                    input_length,
#                                                                    label_length])

# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

# model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)


out = Dense(fc_size, name='final_out')(y_pred)
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
model3 = Model(inputs=input_data, outputs=out)

model3.compile(loss='mean_squared_error',
               optimizer=sgd)

###################################################################


print("Convert Model")
coreml_model = coremltools.converters.keras.convert(model3)

# Set model metadata
coreml_model.author = 'Rob Smith'
coreml_model.license = 'BSD'
coreml_model.short_description = 'Performs keras ds '

# Set feature descriptions manually
coreml_model.input_description['input1'] = 'Audio input'

# Set the output descriptions
coreml_model.output_description['output1'] = 'Audio transcription'

# SAVE
coreml_model.save('kds.mlmodel')

