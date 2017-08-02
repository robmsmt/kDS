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
from keras.layers import Dense, Activation, Bidirectional, Reshape, Lambda, Input, Dropout
from keras.layers.recurrent import _time_distributed_dense
from keras.optimizers import SGD, adam
from keras.preprocessing.sequence import pad_sequences
# from keras.utils.data_utils import Sequence
from keras.layers import TimeDistributed
from keras.layers.merge import add, concatenate
import keras.callbacks

print(keras.__version__) ##be careful with 2.0.6 as 2.0.4 tested with CoreML

from sklearn.utils import shuffle


sys.path.append('./utils/')
from utils import clean, read_text, text_to_int_sequence, int_to_text_sequence
from char_map import char_map, index_map


#######################################################

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

        ## Load file
        #         fs, audio = wav.read(full_wav)
        #         mfcc = p.mfcc(audio, samplerate=fs, numcep=26) # produces time x cep

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

al = {'wavs': train_list_wavs+valid_list_wavs+test_list_wavs,
     'fin': train_list_fin+valid_list_fin+test_list_fin,
     'trans': train_list_trans+valid_list_trans+test_list_trans}

df_all = pd.DataFrame(al, columns=['fin', 'trans', 'wavs'], dtype=int)
df_train = pd.DataFrame(a, columns=['fin', 'trans', 'wavs'], dtype=int)
df_valid = pd.DataFrame(b, columns=['fin', 'trans', 'wavs'], dtype=int)
df_test = pd.DataFrame(c, columns=['fin', 'trans', 'wavs'], dtype=int)

sortagrad = True
if sortagrad:
    df_all = df_all.sort_values(by='fin', ascending=True)
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

print(num_classes)


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
        #             print("1. X_data.shape:",X_data.shape)  # ('1. X_data.shape:', (2, 778, 26))

        assert (X_data.shape == (self.batch_size, 778, 26))

        labels = np.array([get_intseq(l) for l in batch_y_trans])
        #             print("2. labels.shape:",labels.shape) # ('2. labels.shape:', (2, 80))

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

    def next_train(self):
        while 1:
            assert (self.batch_size <= len(self.wavpath))
            if (self.cur_train_index + 1) * self.batch_size >= len(self.wavpath) - self.batch_size:
                self.cur_train_index = 0
                self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                                     self.transcript,
                                                                     self.finish)

            ret = self.get_batch(self.cur_train_index)
            self.cur_train_index += 1

            yield ret

    def next_val(self):
        while 1:
            assert (self.batch_size <= len(self.wavpath))
            if (self.cur_val_index + 1) * self.batch_size >= len(self.wavpath) - self.batch_size:
                self.cur_val_index = 0
                self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                                     self.transcript,
                                                                     self.finish)
            ret = self.get_batch(self.cur_val_index)
            self.cur_val_index += 1

            yield ret

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

    # def on_train_begin(self, logs={}):
    #     # print("train begin")
    #     pass
    #
    # def on_epoch_begin(self, epochs, logs={}):
    #     # print("on epoch begin")
    #     pass

    def on_epoch_end(self, epoch, logs={}):
        print("EPOCH END - shuffling data")
        self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                             self.transcript,
                                                             self.finish)


## todo replace with greedy/beam search

def decode_batch(test_func, word_batch):
   out = test_func([word_batch])[0]
   ret = []
   for j in range(out.shape[0]):

       out_best = list(np.argmax(out[j,:],1))
       out_best = [k for k,g in itertools.groupby(out_best)]

       outStr = int_to_text_sequence(out_best)
       ret.append(''.join(outStr))

   #print(ret)
   return ret


class VizCallback(keras.callbacks.Callback):
    def __init__(self, test_func, validdata_next_val):
        self.test_func = test_func
        self.validdata_next_val = validdata_next_val

    def wer(self):
        pass

    def show_edit_distance_batch(self):
        mean_norm_ed = 0.0
        mean_ed = 0.0

        word_batch = next(self.validdata_next_val)[0]
        #num_proc = batch_size #min of batchsize OR num_left
        decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:batch_size])

        for j in range(0, batch_size):
            edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
            mean_ed += float(edit_dist)
            mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            print("\n{}.Truth:{}\n{}.Trans:{}".format(str(j), word_batch['source_str'][j], str(j), decoded_res[j]))

        mean_norm_ed = mean_norm_ed
        mean_ed = mean_ed

        print('\nOut of %d batch samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (1, mean_ed, mean_norm_ed))


    def on_epoch_end(self, epoch, logs=None):
        # save weights
        self.show_edit_distance_batch()
        #word_batch = next(self.validdata_next_val)[0]
        #result = decode_batch(self.test_func, word_batch['the_input'][0])
        #print("Truth: {} \nTranscribed: {}".format(word_batch['source_str'], result[0]))

# sort_all_fin_list = df_all['fin'].tolist()
# sort_all_trans_list = df_all['trans'].tolist()
# sort_all_wav_list = df_all['wavs'].tolist()

sort_train_fin_list = df_train['fin'].tolist()
sort_train_trans_list = df_train['trans'].tolist()
sort_train_wav_list = df_train['wavs'].tolist()

sort_valid_fin_list = df_valid['fin'].tolist()
sort_valid_trans_list = df_valid['trans'].tolist()
sort_valid_wav_list = df_valid['wavs'].tolist()

sort_test_fin_list = df_test['fin'].tolist()
sort_test_trans_list = df_test['trans'].tolist()
sort_test_wav_list = df_test['wavs'].tolist()

# alldata = timitWavSeq(wavpath=sort_all_wav_list[:32], transcript=sort_all_trans_list[:32], finish=sort_all_fin_list[:32])
# alldata2 = timitWavSeq(wavpath=sort_all_wav_list[:32], transcript=sort_all_trans_list[:32], finish=sort_all_fin_list[:32])
traindata = timitWavSeq(wavpath=sort_train_wav_list, transcript=sort_train_trans_list, finish=sort_train_fin_list)
validdata = timitWavSeq(wavpath=sort_valid_wav_list, transcript=sort_valid_trans_list, finish=sort_valid_fin_list)
testdata = timitWavSeq(wavpath=sort_test_wav_list, transcript=sort_test_trans_list, finish=sort_test_fin_list)


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    print(labels.shape)  # (?, 80)
    print(y_pred.shape)  # (?, 778, 2048)
    print(input_length.shape)  # (?, 1)
    print(label_length.shape)  # (?, 1)

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:

    # y_pred = y_pred[:, 2:, :] ## want to change this?

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Network Params
fc_size = 2048
rnn_size = 512
mfcc_features = 26
max_mfcclength_audio = 778
dropout = [0.2, 0.5, 0.3] ## initial / mid / end
# K.set_learning_phase(1)

X = np.zeros([batch_size, max_mfcclength_audio, mfcc_features])

# Creates a tensor there are always 26 MFCC
input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)
# dr_input = Dropout(dropout[0])(input_data)

# First 3 FC layers
x = Dense(fc_size, name='fc1', activation='relu')(input_data) # >>(?, 778, 2048)
# x = Dropout(dropout[1])(x)
x = Dense(fc_size, name='fc2', activation='relu')(x) # >>(?, 778, 2048)
# x = Dropout(dropout[1])(x)
x = Dense(fc_size, name='fc3', activation='relu')(x) # >>(?, 778, 2048)
# x = Dropout(dropout[1])(x)

# Layer 4 BiDirectional RNN

rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', name='rnn_f')(x) #>>(?, ?, 512) , dropout=dropout[1]

rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
                   kernel_initializer='he_normal', name='rnn_b')(x) #>>(?, ?, 512) , dropout=dropout[1]

#rnn_merged = add([rnn_1f, rnn_1b]) #>>(?, ?, 512)

#TODO TRY THIS FROM: https://github.com/fchollet/keras/issues/2838
rnn_bidir = concatenate([rnn_1f, rnn_1b])
# dr_rnn_bidir = Dropout(dropout[2])(rnn_bidir)
y_pred = TimeDistributed(Dense(num_classes, activation='softmax'))(rnn_bidir)

# Layer 5 FC Layer
#y_pred = Dense(fc_size, name='fc5', activation='relu')(x) #>>(?, 778, 2048)


# Change shape
labels = Input(name='the_labels', shape=[80], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                   labels,
                                                                   input_length,
                                                                   label_length])

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

print(model.summary(line_length=80))

## Make it smaller for perpose of demo
# all_steps = len(sort_all_wav_list)//batch_size
train_steps = len(train_list_wavs)//batch_size
valid_steps = (len(valid_list_wavs)//batch_size)//2

print(train_steps, valid_steps)



# iterate = K.function([input_img, K.learning_phase()], [loss, grads])
test_func = K.function([input_data],[y_pred])

viz_cb = VizCallback(test_func, validdata.next_val())

model.fit_generator(generator=traindata.next_train(),
                    steps_per_epoch=train_steps,  # 28
                    epochs=100,
                    callbacks=[viz_cb, traindata, validdata],  ##create custom callback to handle stop for valid

                    validation_data=validdata.next_val(),
                    validation_steps=1,
                    initial_epoch=0)

model.predict_generator(testdata.next_test(), 8, workers=1, verbose=1)

#
# # serialize model to JSON
with open("ds_ctc_model.json", "w") as json_file:
     json_file.write(model.to_json())
#
# # serialize weights to HDF5
model.save_weights("ds_ctc_model_weights.h5")

