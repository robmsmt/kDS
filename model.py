from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.regularizers import l2

'''
This file builds the models


'''

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape, Lambda, Input,\
    Masking, Convolution1D, BatchNormalization, GRU, Conv1D
from keras.optimizers import SGD, adam
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add  # , # concatenate BAD FOR COREML
from keras.utils.conv_utils import conv_output_length


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_decode(args):

    y_pred, input_length =args

    print(y_pred.shape)
    print(input_length.shape)

    seq_len = tf.squeeze(input_length,axis=1)

    return K.ctc_decode(y_pred=y_pred, input_length=seq_len, greedy=True, beam_width=100, top_paths=1)

def ctc_dec_output(input_shape):
    return (input_shape[0])

def ctc(y_true, y_pred):
    return y_pred

######################################

######################################

def ds1_dropout(fc_size=2048, rnn_size=512, mfcc_features=26,
                         dropout=[0.2, 0.5, 0.3],
                         num_classes=29, train_mode=1):
    '''
    This function builds a neural network as close to the original DS1 paper
    https://arxiv.org/abs/1412.5567

    :param fc_size: The fully connected layer size.
    :param rnn_size: The BIRNN layer size
    :param mfcc_features: Number of MFCC features used
    :param dropout: Uses Dropout values on initial / mid / end
    :param num_classes: Number of output classes (characters to predict, requires 26alpha + apost + space + CTC)
    :return: model, input_data, y_pred (input_data, y_pred used for callbacks)
    '''
    K.set_learning_phase(train_mode)

    # Creates a tensor there are usually 26 MFCC
    input_data = Input(name='the_input', shape=(None, mfcc_features))  # >>(?, max_batch_seq, 26)
    # in_mask = Masking(name='in_mask', mask_value=0)(input_data)
    dr_input = Dropout(dropout[0])(input_data)

    # First 3 FC layers
    x = Dense(fc_size, name='fc1', activation='relu')(dr_input)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)
    x = Dense(fc_size, name='fc2', activation='relu')(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)
    x = Dense(fc_size, name='fc3', activation='relu')(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)

    # Layer 4 BiDirectional RNN
    rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                       kernel_initializer='he_normal', name='rnn_f')(x)  # >>(?, ?, 512) ,

    rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
                       kernel_initializer='he_normal', name='rnn_b')(x)  # >>(?, ?, 512) ,

    rnn_bidir = add([rnn_1f, rnn_1b])
    dr_rnn_bidir = Dropout(dropout[2])(rnn_bidir)

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax"))(dr_rnn_bidir)

    # Change shape
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    return model, input_data, y_pred


def ds1(fc_size=2048, rnn_size=512, mfcc_features=26, num_classes=29):


    input_data = Input(name='the_input', shape=(None,mfcc_features))  # >>(?, 778, 26)


    # First 3 FC layers
    x = TimeDistributed(Dense(fc_size, name='fc1', activation='relu'))(input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc2', activation='relu'))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc3', activation='relu'))(x)  # >>(?, 778, 2048)

    # Layer 4 BiDirectional RNN
    # rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
    #                    kernel_initializer='he_normal', name='rnn_f')(x)  # >>(?, ?, 512) ,
    #
    # rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
    #                    kernel_initializer='he_normal', name='rnn_b')(x)  # >>(?, ?, 512) ,
    # rnn_bidir = add([rnn_1f, rnn_1b])

    x = Bidirectional(SimpleRNN(rnn_size, return_sequences=True, activation='relu', kernel_initializer='he_normal'),
                      merge_mode='sum')(x)

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax"))(x)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                        label_length])


    # dec = Lambda(ctc_decode, output_shape=[None,], name='decoder')([y_pred,input_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    return model, input_data, y_pred, input_length



def build_ds1_simple_rnn_no_ctc_and_xfer_weights(loaded_model, fc_size=2048, rnn_size=512, mfcc_features=26,
                         dropout=[0, 0, 0],
                         num_classes=29):
    '''
    DS1 model but convert into CoreML
    '''

    K.set_learning_phase(0)

    for ind, i in enumerate(loaded_model.layers):
        print(ind, i)

    input_data = Input(name='the_input', shape=(None,mfcc_features))  # >>(?, 778, 26)

    # First 3 FC layers
    x = TimeDistributed(Dense(fc_size, name='fc1', activation='relu',
                              weights=loaded_model.layers[1].get_weights()))(input_data)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc2', activation='relu',
                              weights=loaded_model.layers[2].get_weights()))(x)  # >>(?, 778, 2048)
    x = TimeDistributed(Dense(fc_size, name='fc3', activation='relu',
                              weights=loaded_model.layers[3].get_weights()))(x)  # >>(?, 778, 2048)

    # Layer 4 BiDirectional RNN
    rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                       kernel_initializer='he_normal', name='rnn_f',
        weights = loaded_model.layers[4].get_weights())(x)  # >>(?, ?, 512) ,

    rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
                       kernel_initializer='he_normal', name='rnn_b',
        weights = loaded_model.layers[5].get_weights())(x)  # >>(?, ?, 512) ,

    # rnn_bidir = concatenate([rnn_1f, rnn_1b]) ### CONCAT DOESN'T WORK IN COREML FFS
    rnn_bidir = add([rnn_1f, rnn_1b])

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax",
        weights = loaded_model.layers[7].get_weights()))(x)

    # Change shape
    # labels = Input(name='the_labels', shape=[80], dtype='float32')
    # input_length = Input(name='input_length', shape=[1], dtype='int64')
    # label_length = Input(name='label_length', shape=[1], dtype='int64')
    # model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    model = Model(inputs=input_data, outputs=y_pred)

    return model, input_data, y_pred



def ds2_gru_model(input_dim=26, output_dim=29, nodes=1024, initialization='glorot_uniform'):
    """ Build a recurrent network (CTC) for speech with GRU units """

    input_data = Input(shape=(None, input_dim), name='the_input')

    # todo error InvalidArgumentError (see above for traceback): sequence_length(0) <= 54
    # output = Conv1D(filters=nodes, kernel_size=conv_context,padding='valid',activation='relu',
    #                 kernel_initializer=initialization, strides=2)(input_data)

    output = TimeDistributed(Dense(nodes, name='fc1', activation='relu'))(input_data)
    output = TimeDistributed(Dense(nodes, name='fc2', activation='relu'))(output)
    output = TimeDistributed(Dense(nodes, name='fc3', activation='relu'))(output)

    rnn_1f = GRU(nodes, activation='relu', return_sequences=True, go_backwards=False,
                 kernel_initializer='he_normal', name='rnn_f')(output)

    rnn_1b = GRU(nodes, activation='relu', return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='rnn_b')(output)

    rnn_bidir = add([rnn_1f, rnn_1b])

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax", init=initialization))(rnn_bidir)

    # labels = K.placeholder(name='the_labels', ndim=1, dtype='int32')
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # model.conv_output_length = lambda x: conv_output_length(x, conv_context, conv_border_mode, conv_stride)

    return model, input_data, y_pred


###############################################################


def decode(inputs, **kwargs):
    """ Decodes a sequence of probabilities choosing the path with highest
    probability of occur

    # Arguments
        is_greedy: if True (default) the greedy decoder will be used;
        otherwise beam search decoder will be used

        if is_greedy is False:
            see the documentation of tf.nn.ctc_beam_search_decoder for more
            options

    # Inputs
        A tuple (y_pred, seq_len) where:
            y_pred is a tensor (N, T, C) where N is the bath size, T is the
            maximum timestep and C is the number of classes (including the
            blank label)
            seq_len is a tensor (N,) that indicates the real number of
            timesteps of each sequence

    # Outputs
        A sparse tensor with the top path decoded sequence

    """

    # Little hack for load_model
    import tensorflow as tf
    is_greedy = kwargs.get('is_greedy', True)
    y_pred, seq_len = inputs

    seq_len = tf.cast(seq_len[:, 0], tf.int32)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])

    if is_greedy:
        decoded = tf.nn.ctc_greedy_decoder(y_pred, seq_len)[0][0]
    else:
        beam_width = kwargs.get('beam_width', 100)
        top_paths = kwargs.get('top_paths', 1)
        merge_repeated = kwargs.get('merge_repeated', True)

        decoded = tf.nn.ctc_beam_search_decoder(y_pred, seq_len, beam_width,
                                                top_paths,
                                                merge_repeated)[0][0]

    return decoded

