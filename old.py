import numpy as np
import matplotlib.pyplot as plt

from IPython.core.display import HTML, display
from IPython.display import display, Audio

import sys
import base64
import struct
import os
import fnmatch
import re

import scipy.io.wavfile as wav
import python_speech_features as p

import pandas as pd

# import librosa

#####################################################

# imports
import tensorflow as tf

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

#######################################################

#datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LDC/timit/"
datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/timit/"
target = datapath + "TIMIT/"

train_list_wavs, train_list_trans, train_list_mfcc, train_list_fin = [], [], [], []
valid_list_wavs, valid_list_trans, valid_list_mfcc, valid_list_fin = [], [], [], []
test_list_wavs, test_list_trans, test_list_mfcc, test_list_fin = [], [], [], []

file_count = 0


# token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
def clean(word):
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new


def read_text(full_wav):
    # need to remove _rif.wav (8chars) then add .TXT
    trans_file = full_wav[:-8] + ".TXT"
    with open(trans_file, "r") as f:
        for line in f:
            split = line.split()
            start = split[0]
            end = split[1]
            t_list = split[2:]
            trans = ""
        # insert cleaned word (lowercase plus removed bad punct)
        for t in t_list:
            trans = trans + ' ' + clean(t)

    return start, end, trans


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

df_train = pd.DataFrame(a, columns=['fin', 'trans', 'wavs'], dtype=int)
df_valid = pd.DataFrame(b, columns=['fin', 'trans', 'wavs'], dtype=int)
df_test = pd.DataFrame(c, columns=['fin', 'trans', 'wavs'], dtype=int)

sortagrad = True
if sortagrad:
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

# From Baidu ba-dls-deepspeech - https://github.com/baidu-research/ba-dls-deepspeech

char_map_str = """
' 1
<SPACE> 2
a 3
b 4
c 5
d 6
e 7
f 8
g 9
h 10
i 11
j 12
k 13
l 14
m 15
n 16
o 17
p 18
q 19
r 20
s 21
t 22
u 23
v 24
w 25
x 26
y 27
z 28
"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[2] = ' '


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


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

num_classes = len(char_map)

print(num_classes)


global batch_size
batch_size = 8


def get_intseq(trans):
    # PAD

    #print(trans)
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

    #         def __len__(self):
    #             ## returns number of batches in the sequence
    #             return len(self.wavpath) // self.batch_size

    def get_batch(self, idx):
        #print(idx)
        #print(self.cur_train_index)

        batch_x = self.wavpath[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_trans = self.transcript[idx * self.batch_size:(idx + 1) * self.batch_size]

        try:
            assert (len(batch_x) == self.batch_size)
            assert (len(batch_y_trans) == self.batch_size)
        except Exception as e:
            print(e)
            print(batch_x)
            print(batch_y_trans)

        X_data = np.array([get_mfcc(file_name) for file_name in batch_x])
        #             print("1. X_data.shape:",X_data.shape)  # ('1. X_data.shape:', (2, 778, 26))

        assert (X_data.shape == (self.batch_size, 778, 26))

        labels = np.array([get_intseq(l) for l in batch_y_trans])
        #             print("2. labels.shape:",labels.shape) # ('2. labels.shape:', (2, 80))

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
            'label_length': label_length
        }

        outputs = {'ctc': np.zeros([batch_size])}

        return (inputs, outputs)

    def next_train(self):
        while 1:
            if (self.cur_train_index+1)*self.batch_size >= len(self.wavpath)-self.batch_size:
                self.cur_train_index = 0

            ret = self.get_batch(self.cur_train_index)
            self.cur_train_index += 1

            yield ret

    def next_val(self):
        while 1:
            if (self.cur_val_index+1)*self.batch_size >= len(self.wavpath)-self.batch_size:
                self.cur_val_index = 0
            ret = self.get_batch(self.cur_val_index)
            self.cur_val_index += 1

            yield ret

    def next_test(self):
        while 1:
            if (self.cur_test_index+1)*self.batch_size >= len(self.wavpath)-self.batch_size:
                self.cur_test_index = 0

            ret = self.get_batch(self.cur_test_index)
            self.cur_test_index += 1

            yield ret

    def on_train_begin(self, logs={}):
        print("train begin")

    def on_epoch_begin(self, epochs, logs={}):
        print("on epoch begin")

    def on_epoch_end(self, epoch, logs={}):
        print("EPOCH END")

sort_train_fin_list = df_train['fin'].tolist()
sort_train_trans_list = df_train['trans'].tolist()
sort_train_wav_list = df_train['wavs'].tolist()

sort_valid_fin_list = df_valid['fin'].tolist()
sort_valid_trans_list = df_valid['trans'].tolist()
sort_valid_wav_list = df_valid['wavs'].tolist()

sort_test_fin_list = df_test['fin'].tolist()
sort_test_trans_list = df_test['trans'].tolist()
sort_test_wav_list = df_test['wavs'].tolist()

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

X = np.zeros([batch_size, max_mfcclength_audio, mfcc_features])

# Creates a tensor there are always 26 MFCC
input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)

# First 3 FC layers
x = Dense(fc_size, name='fc1', activation='relu')(input_data) # >>(?, 778, 2048)
x = Dense(fc_size, name='fc2', activation='relu')(x) # >>(?, 778, 2048)
x = Dense(fc_size, name='fc3', activation='relu')(x) # >>(?, 778, 2048)

# Layer 4 BiDirectional RNN

rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', name='rnn_f')(x) #>>(?, ?, 512)

rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                   kernel_initializer='he_normal', name='rnn_b')(x) #>>(?, ?, 512)

rnn_merged = add([rnn_1f, rnn_1b]) #>>(?, ?, 512)

#TODO TRY THIS FROM: https://github.com/fchollet/keras/issues/2838
# rnn_bidir1 = merge([rnn_fwd1, rnn_bwd1], mode='concat')
# predictions = TimeDistributed(Dense(output_class_size, activation='softmax'))(rnn_bidir1)

x = Activation('relu', name='birelu')(rnn_merged) #>>(?, ?, 512)

# Layer 5 FC Layer
y_pred = Dense(fc_size, name='fc5', activation='relu')(x) #>>(?, 778, 2048)


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
train_steps = len(train_list_wavs)//batch_size
valid_steps = len(valid_list_wavs)//batch_size

print(train_steps, valid_steps)

model.fit_generator(generator=traindata.next_train(),
                    steps_per_epoch=train_steps,  # 28
                    epochs=4,
                    callbacks=[traindata],  ##create custom callback to handle stop for valid

                    validation_data=None,
                    validation_steps=None,
                    initial_epoch=0)

model.predict_generator( testdata.next_test(), 5, workers=1, verbose=1)

#
# # serialize model to JSON
# with open("ds_ctc_model.json", "w") as json_file:
#     json_file.write(model.to_json())
#
# # serialize weights to HDF5
# model.save_weights("ds_ctc_model_weights.h5")

# # save data to .npz
# np.savez('xor_data.npz', training_data=training_data, target_data=target_data, test_data=test_data)

#
# ##########################################
#
# import coremltools
# from keras.models import model_from_json
#
# # load json and create model
# json_file = open('ds_ctc_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# # load weights into loaded model
# loaded_model.load_weights("ds_ctc_model_weights.h5")
# print("Loaded model/weights from disk")
#
# # it looks like this has worked. We can now convert to
# print("Convert Model")
# try:
#     coreml_model = coremltools.converters.keras.convert(loaded_model)
# except Exception as e:
#     print(e)
#
#     # # Set model metadata
#     # coreml_model.author = 'Rob Smith'
#     # coreml_model.license = 'BSD'
#     # coreml_model.short_description = 'Performs keras ds ctc '
#
#     # # SAVE
#     # coreml_model.save('kds_ctc.mlmodel')
#
#
# # Network Params
# fc_size = 2048
# rnn_size = 512
# mfcc_features = 26
# max_mfcclength_audio = 778
#
# X = np.zeros([batch_size, max_mfcclength_audio, mfcc_features])
#
# # Creates a tensor there are always 26 MFCC
# input_data = Input(name='the_input', shape=X.shape[1:]) # >>(?, 778, 26)
#
# # First 3 FC layers
# x = Dense(fc_size, name='fc1', activation='relu',
#           weights=loaded_model.layers[1].get_weights())(input_data) # >>(?, 778, 2048)
# x = Dense(fc_size, name='fc2', activation='relu',
#          weights=loaded_model.layers[2].get_weights())(x) # >>(?, 778, 2048)
# x = Dense(fc_size, name='fc3', activation='relu',
#          weights=loaded_model.layers[3].get_weights())(x) # >>(?, 778, 2048)
#
# # Layer 4 BiDirectional RNN
#
# rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
#                    kernel_initializer='he_normal', name='rnn_f',
#                    weights=loaded_model.layers[4].get_weights())(x) #>>(?, ?, 512)
#
# rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
#                    kernel_initializer='he_normal', name='rnn_b',
#                    weights=loaded_model.layers[5].get_weights())(x) #>>(?, ?, 512)
#
# rnn_merged = add([rnn_1f, rnn_1b],
#                 weights=loaded_model.layers[6].get_weights()) #>>(?, ?, 512)
# x = Activation('relu', name='birelu',
#               weights=loaded_model.layers[7].get_weights())(rnn_merged) #>>(?, ?, 512)
#
# # Layer 5 FC Layer
# y_pred = Dense(fc_size, name='fc5', activation='relu',
#               weights=loaded_model.layers[8].get_weights())(x) #>>(?, 778, 2048)
#
# ##############################################
#
#
# #####################################
#
# # Change shape
# labels = Input(name='the_labels', shape=[80], dtype='float32')
# input_length = Input(name='input_length', shape=[1], dtype='int64')
# label_length = Input(name='label_length', shape=[1], dtype='int64')
#
# # Keras doesn't currently support loss funcs with extra parameters
# # so CTC loss is implemented in a lambda layer
# # loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
# #                                                                    labels,
# #                                                                    input_length,
# #                                                                    label_length])
#
# # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#
# # model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
#
# # # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
# # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
#
#
# out = Dense(fc_size, name='final_out')(y_pred)
# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# model3 = Model(inputs=input_data, outputs=out)
#
# model3.compile(loss='mean_squared_error',
#                optimizer=sgd)
#
# ###################################################################
#
#
# print("Convert Model")
# coreml_model = coremltools.converters.keras.convert(model3)
#
# # Set model metadata
# coreml_model.author = 'Rob Smith'
# coreml_model.license = 'BSD'
# coreml_model.short_description = 'Performs keras ds '
#
# # Set feature descriptions manually
# #coreml_model.input_description['the_input'] = 'Audio input'
# # coreml_model.input_description['bathrooms'] = 'Number of bathrooms'
# # coreml_model.input_description['size'] = 'Size (in square feet)'
#
# # Set the output descriptions
# # coreml_model.output_description['out'] = 'Audio transcription'
#
# # SAVE
# coreml_model.save('kds.mlmodel')
#
