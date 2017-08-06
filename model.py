
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
                         num_classes=29, train_mode=1):
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
    K.set_learning_phase(train_mode)

    # Creates a tensor there are always 26 MFCC
    # X = np.zeros([16, max_mfcclength_audio, mfcc_features])

    #input_data = Input(name='the_input', shape=X.shape[1:])  # >>(?, 778, 26)
    input_data = Input(name='the_input', shape=(None,26))  # >>(?, 778, 26)
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

    # rnn_bidir = concatenate([rnn_1f, rnn_1b]) ### CONCAT DOESN'T WORK IN COREML FFS
    rnn_bidir = add([rnn_1f, rnn_1b])
    dr_rnn_bidir = Dropout(dropout[2])(rnn_bidir)

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax"))(dr_rnn_bidir)

    # Change shape
    labels = Input(name='the_labels', shape=[None,], dtype='float32')
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

    # dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    print(model.summary(line_length=80))

    return model, input_data, y_pred


def build_ds1_simple_rnn_no_ctc_and_xfer_weights(loaded_model, fc_size=2048, rnn_size=512, mfcc_features=26,
                         max_mfcclength_audio=778, dropout=[0, 0, 0],
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
    K.set_learning_phase(0)

    for ind, i in enumerate(loaded_model.layers):
        print(ind, i)

    # Creates a tensor there are always 26 MFCC
    X = np.zeros([16, max_mfcclength_audio, mfcc_features])

    input_data = Input(name='the_input', shape=X.shape[1:])  # >>(?, 778, 26)
    # in_mask = Masking(name='in_mask', mask_value=0)(input_data) ## DOESN'T WORK IN COREML FFS
    dr_input = Dropout(dropout[0])(input_data)

    # First 3 FC layers
    x = Dense(fc_size, name='fc1', activation='relu',
        weights = loaded_model.layers[3].get_weights())(dr_input)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)
    x = Dense(fc_size, name='fc2', activation='relu',
        weights = loaded_model.layers[5].get_weights())(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)
    x = Dense(fc_size, name='fc3', activation='relu',
        weights = loaded_model.layers[7].get_weights())(x)  # >>(?, 778, 2048)
    x = Dropout(dropout[1])(x)

    # Layer 4 BiDirectional RNN

    rnn_1f = SimpleRNN(rnn_size, return_sequences=True, go_backwards=False,
                       kernel_initializer='he_normal', name='rnn_f',
        weights = loaded_model.layers[9].get_weights())(x)  # >>(?, ?, 512) ,

    rnn_1b = SimpleRNN(rnn_size, return_sequences=True, go_backwards=True,
                       kernel_initializer='he_normal', name='rnn_b',
        weights = loaded_model.layers[10].get_weights())(x)  # >>(?, ?, 512) ,

    # rnn_bidir = concatenate([rnn_1f, rnn_1b]) ### CONCAT DOESN'T WORK IN COREML FFS
    rnn_bidir = add([rnn_1f, rnn_1b])
    dr_rnn_bidir = Dropout(dropout[2])(rnn_bidir)

    # Layer 5+6 Time Dist Layer & Softmax
    y_pred = TimeDistributed(Dense(num_classes, name="y_pred", activation="softmax",
        weights = loaded_model.layers[13].get_weights()))(dr_rnn_bidir)

    # Change shape
    # labels = Input(name='the_labels', shape=[80], dtype='float32')
    # input_length = Input(name='input_length', shape=[1], dtype='int64')
    # label_length = Input(name='label_length', shape=[1], dtype='int64')
    # model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=input_data, outputs=y_pred)

    model.compile(loss='mean_squared_error', optimizer=sgd)  ## try MSE

    print(model.summary(line_length=80))

    return model, input_data, y_pred

def conv_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
       time. Note that this function is in line with the function used in
       Convolution1D class from Keras.
    Params:
       input_length (int): Length of the input sequence.
       filter_size (int): Width of the convolution kernel.
       border_mode (str): Only support `same` or `valid`.
       stride (int): Stride size used in 1D convolution.
       dilation (int)
    """
    if input_length is None:
       return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
       output_length = input_length
    elif border_mode == 'valid':
       output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride



def ds2_gru_model(input_dim=26, output_dim=30, nodes=1024,
                     conv_context=11, conv_border_mode='valid', conv_stride=2,
                     initialization='glorot_uniform', batch_norm=False):
    """ Build a recurrent network (CTC) for speech with GRU units """


    input_data = Input(shape=(None, input_dim), name='the_input')

    # todo error InvalidArgumentError (see above for traceback): sequence_length(0) <= 54
    # output = Conv1D(filters=nodes, kernel_size=conv_context,padding='valid',activation='relu',
    #                 kernel_initializer=initialization, strides=2)(input_data)

    output = Dense(nodes, name='fc1', activation='relu')(input_data)
    output = Dense(nodes, name='fc2', activation='relu')(output)
    output = Dense(nodes, name='fc3', activation='relu')(output)

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

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # model.conv_output_length = lambda x: conv_output_length(x, conv_context, conv_border_mode, conv_stride)
    # dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    print(model.summary(line_length=80))

    return model, input_data, y_pred

