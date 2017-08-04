#!/usr/bin/env python

'''

KERAS Deep Speech - end to end speech recognition. Designed for
use with CoreML 0.4 to use model on iOS

see conversion scripts etc

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



def main(sortagrad, loadcheckpoint, epochs, batchsize):
    '''
    There are 5 simple steps to this program
    '''

    ## 1. get path and data
    path = get_data_path()
    dataproperties, df_all, df_train, df_valid, df_test = get_all_wavs_in_path(path, sortagrad=sortagrad)

    ## 2. init data generators
    alldata = BaseGenerator(dataframe=df_all, dataproperties=dataproperties, batch_size=batchsize)
    traindata = BaseGenerator(dataframe=df_train, dataproperties=dataproperties, batch_size=batchsize)
    validdata = BaseGenerator(dataframe=df_valid, dataproperties=dataproperties, batch_size=batchsize)
    testdata = BaseGenerator(dataframe=df_test, dataproperties=dataproperties, batch_size=batchsize)


    ## 3. Load existing or create new model
    if loadcheckpoint:
        # load existing

        _, input_data, y_pred = build_ds1_simple_rnn()  # required for callback todo test
        model = load_model_checkpoint()
    else:
        # new model

        model, input_data, y_pred = build_ds1_simple_rnn(fc_size=2048,
                                                         rnn_size=512,
                                                         mfcc_features=26,
                                                         max_mfcclength_audio=778,
                                                         dropout=[0.2, 0.5, 0.3],
                                                         num_classes=30)

    ## 4. train
    all_steps = len(df_train.index) // batchsize
    train_steps = len(df_train.index) // batchsize
    valid_steps = (len(df_valid.index) // batchsize) // 10

    if socket.gethostname().lower() in 'rs-e5550'.lower(): train_steps = 2

    iterate = K.function([input_data, K.learning_phase()], [y_pred])
    test_cb = TestCallback(iterate, validdata.next_batch())

    cp_cb = ModelCheckpoint(filepath='./checkpoints/epoch_checkpoint.hdf5', verbose=1, save_best_only=False)
    tb_cb = TensorBoard(log_dir='./tensorboard/', histogram_freq=1, write_graph=True, write_images=True)

    model.fit_generator(generator=traindata.next_batch(),
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        callbacks=[cp_cb, tb_cb, test_cb, traindata, validdata],  ##create custom callback to handle stop for valid
                        validation_data=validdata.next_batch(),
                        validation_steps=1,
                        initial_epoch=0
                        )

    ## 5. final test - move this to run-test
    model.predict_generator(testdata.next_batch(), 8, workers=1, verbose=1)

    ## save final version of model
    save_model(model, name="./checkpoints/ds_ctc_FIN")



#######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--loadcheckpoint', type=bool, default=False,
                       help='If true, load for the last checkpoint at the default path we find')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=16,
                       help='batch_size used to train the model')

    args = parser.parse_args()

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.sortagrad, args.loadcheckpoint, args.epochs, args.batchsize)


###
#todo this is not a ready output
#testdata.export_test_mfcc()


