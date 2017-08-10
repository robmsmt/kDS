#!/usr/bin/env python

'''

KERAS Deep Speech - test script

'''

#####################################################

import argparse
import datetime

#####################################################

from utils import *
from generator import *
from data import *
from model import *

import keras

# from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

#######################################################


def main(checkpointpath, runtimestr):
    '''
    There are 5 simple steps to this program
    '''

    ## 1. get path and data
    path = get_timit_data_path()
    dataproperties, df_all, df_train, df_valid, df_test = get_all_wavs_in_path(path, sortagrad=False)

    # as we are testing we will merge df_test + df_valid to give one complete set
    frames = [df_valid, df_test]
    df_supertest = pd.concat(frames)

    ## 2. init data generators
    testdata = BaseGenerator(dataframe=df_supertest, dataproperties=dataproperties, batch_size=2)


    ## 3. Load existing or error
    if checkpointpath:
        # load existing

        fresh_model, input_data, y_pred = ds1(fc_size=2048, rnn_size=512,
                                              mfcc_features=26, num_classes=29)

        model = load_model_checkpoint(checkpointpath)
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss='mean_squared_error', optimizer=sgd)  ## try MSE


    else:
        # new model
        raise("You need to load an existing trained model")



    ## 4. test

    test_steps = len(df_supertest.index)
    print(test_steps)

    if socket.gethostname().lower() in 'rs-e5550'.lower(): test_steps = 2

    ## 5. PREDICT (same as iOS?)
    # predictions = model.predict_generator(testdata.next_batch(), test_steps, workers=1, verbose=1)
    # pred1 = test_decode(predictions, batch_size=test_steps)
    # # todo fix bug-  this doesn't work well, is it trained enough? Howcome valid test works well- let's use that for now
    # print(pred1)

    d1 = np.loadtxt('./test_mfccs/new/test_mfcc_0.csv', dtype=float, delimiter=",")
    d1 = np.expand_dims(d1, axis=0)
    print(d1.shape)
    pred = model.predict(d1)
    print(pred)
    print(pred.shape)
    np.savetxt('./test_mfccs/new/test_output.csv', pred[0, :, :], delimiter=',')

    ## 5. EVALUATE (same as iOS?)

    # out = [ 6, 17, 16,  1, 22,  2,  3, 21, 13,  2, 15,  7,  2, 22, 17,  2,  5,  3, 20, 20, 27,  2,  3, 16,
    #         2, 17, 11, 14, 27,  2, 20,  3,  9,  2, 14, 11, 13,  7,  2, 22, 10,  3, 22]
    #
    # pred2 = model.evaluate(d1, out)
    # print(pred2)

    ## 5. CUSTOM
    # pretend that epoch is finished and exec

    iterate = K.function([input_data, K.learning_phase()], [y_pred])
    test_cb = TestCallback(iterate, testdata, model=model, runtimestr=runtimestr)

    res = test_cb.validate_epoch_end()

    print(test_cb.mean_wer_log)


#######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    # parser.add_argument('--fullcheckpointpath', type=str, default="./checkpoints/TRIMMED_ds_model",
    parser.add_argument('--checkpointpath', type=str, default="./checkpoints/trimmed/TRIMMED_ds_model",
                        help='load checkpoint at this path')


    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.checkpointpath, runtime)


###
#todo this is not a ready output
#testdata.export_test_mfcc()


