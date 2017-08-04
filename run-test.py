#!/usr/bin/env python

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


hostname = socket.gethostname().lower()

## Use hostname to detect my laptop OR else it's cluster
if hostname in ('rs-e5550').lower():
    datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LDC/timit/"
else:
    datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/timit/"

target = datapath + "TIMIT/"

train_list_wavs, train_list_trans, train_list_mfcc, train_list_fin = [], [], [], []
valid_list_wavs, valid_list_trans, valid_list_mfcc, valid_list_fin = [], [], [], []
test_list_wavs, test_list_trans, test_list_mfcc, test_list_fin = [], [], [], []

file_count = 0


for root, dirnames, filenames in os.walk(target):
    for filename in fnmatch.filter(filenames, "*.wav"):

        full_wav = os.path.join(root, filename)

        _, end, trans = read_text(full_wav)

        if 'train' in full_wav.lower():
            train_list_wavs.append(full_wav)
            train_list_trans.append(trans)
            train_list_fin.append(end)

        elif 'test' in full_wav.lower():
            ##split 50/50 into validation and test (note not random)
            if file_count % 2 == 0:
                test_list_wavs.append(full_wav)
                test_list_trans.append(trans)
                test_list_fin.append(end)
            else:
                valid_list_wavs.append(full_wav)
                valid_list_trans.append(trans)
                valid_list_fin.append(end)
        else:
            raise IOError

        file_count = file_count + 1

a = {'wavs': train_list_wavs,
     'fin': train_list_fin,
     'trans': train_list_trans}

b = {'wavs': valid_list_wavs,
     'fin': valid_list_fin,
     'trans': valid_list_trans}

c = {'wavs': test_list_wavs,
     'fin': test_list_fin,
     'trans': test_list_trans}
#
# al = {'wavs': train_list_wavs+valid_list_wavs+test_list_wavs,
#      'fin': train_list_fin+valid_list_fin+test_list_fin,
#      'trans': train_list_trans+valid_list_trans+test_list_trans}
#
# df_all = pd.DataFrame(al, columns=['fin', 'trans', 'wavs'], dtype=int)
df_train = pd.DataFrame(a, columns=['fin', 'trans', 'wavs'], dtype=int)
df_valid = pd.DataFrame(b, columns=['fin', 'trans', 'wavs'], dtype=int)
df_test = pd.DataFrame(c, columns=['fin', 'trans', 'wavs'], dtype=int)

sortagrad = True
if sortagrad:
    # df_all = df_all.sort_values(by='fin', ascending=True)
    df_train = df_train.sort_values(by='fin', ascending=True)
    df_valid = df_valid.sort_values(by='fin', ascending=True)
    df_test = df_test.sort_values(by='fin', ascending=True)

print(len(train_list_wavs) + len(test_list_wavs) + len(valid_list_wavs))
print(len(train_list_wavs), len(test_list_wavs), len(valid_list_wavs))
# 6300
# (4620, 840, 840)

## looks like index 150 is the longest sentence with 80chars so we'll have to pad on this.
max_mfcc_index = 0
max_mfcc_len = 0
max_trans_charlength = 0
all_words = []

comb = train_list_trans + test_list_trans + valid_list_trans
# comb_mfcc = train_list_mfcc+test_list_mfcc+valid_list_mfcc

for count, sent in enumerate(comb):
    # count length
    if len(sent) > max_trans_charlength:
        max_trans_charlength = len(sent)
    # build vocab
    for w in sent.split():
        all_words.append(clean(w))
        # check mfcc
# if(comb_mfcc[count].shape[0]>max_mfcc_len):
#         max_mfcc_len=comb_mfcc[count].shape[0]
#         max_mfcc_index=count

print("max_trans_charlength:", max_trans_charlength)
# print("max_mfcc_len:",max_mfcc_len, "at comb index:",max_mfcc_index)

# ('max_trans_charlength:', 80)
# ('max_mfcc_len:', 778, 'at comb index:', 541)

all_vocab = set(all_words)
## do some analysis here on the types of words. E.g. a ? will change the sound of a word a lot.
print("Words:",len(all_words))
print("Vocab:",len(all_vocab))



##CHECK
max_intseq_length = 0
for x in train_list_trans + test_list_trans + valid_list_trans:
    try:
        y = text_to_int_sequence(x)
        if len(y) > max_intseq_length:
            max_intseq_length = len(y)
    except:
        print("error at:", x)

print(max_intseq_length)


num_classes = len(char_map)+2 ##need +2 for ctc null char

print("numclasses:",num_classes)


global batch_size
batch_size = 16


def get_intseq(trans):
    # PAD
    while (len(trans) < max_intseq_length):
        trans = trans + ' '  # replace with a space char to pad
    t = text_to_int_sequence(trans)
    return t


def get_mfcc(filename):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=778, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}


def get_xsize(val):
    return val.shape[0]


def get_ylen(val):
    return len(val)


class timitWavSeq(keras.callbacks.Callback):
    def __init__(self, wavpath, transcript, finish):

        self.wavpath = wavpath
        self.transcript = transcript
        self.start = np.zeros(len(finish))
        self.finish = finish
        self.length = self.finish
        self.batch_size = batch_size
        self.cur_train_index = 0
        self.cur_val_index = 0
        self.cur_test_index = 0

        # print(self.transcript)

    #         def __len__(self):
    #             ## returns number of batches in the sequence
    #             return len(self.wavpath) // self.batch_size

    def get_batch(self, idx):

        batch_x = self.wavpath[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_trans = self.transcript[idx * self.batch_size:(idx + 1) * self.batch_size]

        try:
            assert (len(batch_x) == self.batch_size)
            assert (len(batch_y_trans) == self.batch_size)
        except Exception as e:
            print(e)
            print(batch_x)
            print(batch_y_trans)

        # source_str = []

        X_data = np.array([get_mfcc(file_name) for file_name in batch_x])

        assert (X_data.shape == (self.batch_size, 778, 26))

        labels = np.array([get_intseq(l) for l in batch_y_trans])
        source_str = np.array([l for l in batch_y_trans])

        assert (labels.shape == (self.batch_size, max_intseq_length))

        input_length = np.array([get_xsize(mfcc) for mfcc in X_data])
        #             print("3. input_length:",input_length.shape) # ('3. input_length:', (2,))
        assert (input_length.shape == (self.batch_size,))

        label_length = np.array([get_ylen(y) for y in labels])
        #             print("4. label_length:",label_length.shape) # ('4. label_length:', (2,))
        assert (label_length.shape == (self.batch_size,))

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str
        }

        outputs = {'ctc': np.zeros([batch_size])}

        return (inputs, outputs)

    def next_test(self):
        while 1:
            assert(self.batch_size<=len(self.wavpath))
            if (self.cur_test_index + 1) * self.batch_size >= len(self.wavpath) - self.batch_size:
                self.cur_test_index = 0
                self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                                     self.transcript,
                                                                     self.finish)

            ret = self.get_batch(self.cur_test_index)
            self.cur_test_index += 1

            yield ret


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):

        ## todo replace with greedy/beam search
        out_best = list(np.argmax(out[j,:],1))
        out_best = [k for k,g in itertools.groupby(out_best)]
        try:
            outStr = int_to_text_sequence(out_best)
        except Exception as e:
            print("error:", e)
            outStr = "DECODE ERROR"

        ret.append(''.join(outStr))

    return ret


sort_test_fin_list = df_test['fin'].tolist()
sort_test_trans_list = df_test['trans'].tolist()
sort_test_wav_list = df_test['wavs'].tolist()

testdata = timitWavSeq(wavpath=sort_test_wav_list, transcript=sort_test_trans_list, finish=sort_test_fin_list)


###############################
trimmed = True

if trimmed:
    # load json and create model
    print("Loading trimmed")
    json_file = open('TRIMMED_ds_ctc_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights("TRIMMED_ds_ctc_model_weights.h5")
    print("Loaded TRIMMED model/weights from disk")
    loaded_model.summary()
else:
    print("Loading normal model")
    json_file = open('ds_ctc_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights("ds_ctc_model_weights.h5")
    print("Loaded TRIMMED model/weights from disk")
    loaded_model.summary()

# K.set_learning_phase(0)
X = np.zeros([1, 778, 26])

# let's load in some data from the test_mfcc folders
# x0 = np.loadtxt('./test_mfccs/mfcc_test_0.csv', dtype=float, delimiter=',')
# y = np.expand_dims(x0, axis=0)

input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)
y_pred = TimeDistributed(Dense(num_classes, activation='softmax'))(input_data)
test_func = K.function([input_data, K.learning_phase()],[y_pred])

##### we are ready to fire the mfcc's into the net

### FIRE


predictions = loaded_model.predict_generator(testdata.next_test(), 1, workers=1, verbose=1)

result = test_func(testdata.next_test())

print(predictions.shape)
print(predictions)
