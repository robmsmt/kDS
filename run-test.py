#!/usr/bin/env python

'''

KERAS Deep Speech - test script

'''

#####################################################

import argparse

#####################################################

from utils import *
from generator import *
from data import *
from model import *

# from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

#######################################################



def main(loadcheckpoint, fullcheckpointpath):
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
    testdata = BaseGenerator(dataframe=df_supertest, dataproperties=dataproperties, batch_size=1)


    ## 3. Load existing or error
    if loadcheckpoint:
        # load existing

        # fresh_model, input_data, y_pred = build_ds1_simple_rnn(fc_size=2048,
        #                                                  rnn_size=512,
        #                                                  mfcc_features=26,
        #                                                  max_mfcclength_audio=778,
        #                                                  dropout=[0.0, 0.0, 0.0],
        #                                                  num_classes=30,
        #                                                  train_mode=0)

        model = load_model_checkpoint(fullcheckpointpath)

    else:
        # new model
        raise("You need to load an existing trained model")



    ## 4. test

    test_steps = len(df_supertest.index)

    if socket.gethostname().lower() in 'rs-e5550'.lower(): test_steps = 20

    # iterate = K.function([input_data, K.learning_phase()], [y_pred])
    # test_cb = TestCallback(iterate, testdata.next_batch())


    ## 5. make predictions
    predictions = model.predict_generator(testdata.next_batch(), test_steps, workers=1, verbose=1)

    r = test_decode(predictions, batch_size=test_steps)
    # todo fix bug-  this doesn't work well, is it trained enough? Howcome valid test works well- let's use that for now
    print(r)

#######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--loadcheckpoint', type=bool, default=True,
                       help='If true, load for the last checkpoint at the default path we find')
    # parser.add_argument('--fullcheckpointpath', type=str, default="./checkpoints/TRIMMED_ds_model",
    parser.add_argument('--fullcheckpointpath', type=str, default="./checkpoints/TRIMMED_ds_model",
                        help='If true, we sort utterances by their length in the first epoch')


    args = parser.parse_args()

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.loadcheckpoint, args.fullcheckpointpath)


###
#todo this is not a ready output
#testdata.export_test_mfcc()


