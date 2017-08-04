
'''
This file builds the models


'''

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape, Lambda, Input, Masking
from keras.optimizers import SGD, adam
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add  # , # concatenate BAD FOR COREML


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    print("CTC lambda inputs / shape")
    print("y_pred:",y_pred.shape)  # (?, 778, 30)
    print("labels:",labels.shape)  # (?, 80)
    print("input_length:",input_length.shape)  # (?, 1)
    print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_ds1_simple_rnn(fc_size=2048, rnn_size=512, mfcc_features=26,
                         max_mfcclength_audio=778, dropout=[0.2, 0.5, 0.3],
                         num_classes=29):
    '''
    This function builds a neural network as close to the original DS1 paper
    https://arxiv.org/abs/1412.5567

    :param fc_size: The fully connected layer size.
    :param rnn_size: The BIRNN layer size
    :param mfcc_features: Number of MFCC features used
    :param max_mfcclenth_audio: Number of
    :param dropout: Uses Dropout values on initial / mid / end
    :param num_classes: Number of output classes (characters to predict, requires 26alpha + apost + space + CTC)
    :return: model, input_data, y_pred (input_data, y_pred used for callbacks)
    '''
    K.set_learning_phase(1)

    # Creates a tensor there are always 26 MFCC
    X = np.zeros([16, max_mfcclength_audio, mfcc_features])

    input_data = Input(name='the_input', shape=X.shape[1:])  # >>(?, 778, 26)
    in_mask = Masking(name='in_mask', mask_value=0)(input_data)
    dr_input = Dropout(dropout[0])(in_mask)

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

    # rnn_bidir = concatenate([rnn_1f, rnn_1b]) ### CONCAT DOESN'T WORK IN COREML FFS
    rnn_bidir = add([rnn_1f, rnn_1b])
    dr_rnn_bidir = Dropout(dropout[2])(rnn_bidir)

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax"))(dr_rnn_bidir)

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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    print(model.summary(line_length=80))

    return model, input_data, y_pred
