#!/usr/bin/env python

'''

KERAS Deep Speech - end to end speech recognition. Designed for
use with CoreML 0.4 to use model on iOS

see conversion scripts etc

'''

#####################################################

import argparse
import datetime

#####################################################

from utils import *
from generator import *
from data import *
from model import *

# from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras

#######################################################



def main(sortagrad, loadcheckpoint, epochs, batchsize, tensorboard, runtime, deepspeech):
    '''
    There are 5 simple steps to this program
    '''

    runtimestr = "DS"+str(deepspeech)+"_"+runtime

    ## 1. get path and data
    timit_path = get_timit_data_path()
    dataproperties, df_all, df_train, df_valid, df_test = get_all_wavs_in_path(timit_path, sortagrad=sortagrad)

    #merge
    frames = [df_valid, df_test]
    df_supertest = pd.concat(frames)


    ## 1b. load in Librispeech
    libri_path = get_librispeech_data_path()

    lib_filelist = ["librivox-dev-clean.csv,", "librivox-dev-other.csv,",
                "librivox-train-clean-100.csv,","librivox-train-clean-360.csv,", "librivox-train-other-500.csv,",
                "librivox-test-clean.csv,","librivox-test-other.csv"]

    if socket.gethostname().lower() in 'rs-e5550'.lower(): lib_filelist=["librivox-dev-clean.csv"]

    # timit_filelist = ["df_all.csv"]

    csvs = ""
    for f in lib_filelist:
        csvs = csvs + libri_path + f

    dataproperties, df_lib_all = check_all_wavs_and_trans_from_csvs(csvs, df_train)

    ## 2. init data generators
    traindata = BaseGenerator(dataframe=df_lib_all, dataproperties=dataproperties, batch_size=batchsize)
    validdata = BaseGenerator(dataframe=df_supertest, dataproperties=dataproperties, batch_size=batchsize)


    ## 3. Load existing or create new model
    if loadcheckpoint:
        # load existing -todo test this

        _, input_data, y_pred = build_ds1_simple_rnn(fc_size=2048,
                                                         rnn_size=512,
                                                         mfcc_features=26,
                                                         max_mfcclength_audio=778,
                                                         dropout=[0.2, 0.5, 0.3],
                                                         num_classes=30,
                                                         train_mode=1)  # required for callback todo test
        model = load_model_checkpoint()

    else:
        # new model

        if(deepspeech==1):
            model, input_data, y_pred = build_ds1_simple_rnn(fc_size=2048,
                                                             rnn_size=512,
                                                             mfcc_features=26,
                                                             max_mfcclength_audio=778,
                                                             dropout=[0.2, 0.5, 0.3],
                                                             num_classes=30,
                                                             train_mode=1)
        elif(deepspeech==2):
            model, input_data, y_pred = ds2_gru_model(input_dim=26, output_dim=30,nodes=1024, conv_context=11,
                                                      conv_border_mode='valid', conv_stride=2,
                                                      initialization='glorot_uniform', batch_norm=False)

        # testmodel, test_input_data, test_y_pred = build_ds1_simple_rnn(fc_size=2048,
        #                                                  rnn_size=512,
        #                                                  mfcc_features=26,
        #                                                  max_mfcclength_audio=778,
        #                                                  dropout=[0.0, 0.0, 0.0],
        #                                                  num_classes=30,
        #                                                  train_mode=0)

    ## 4. train
    train_steps = len(df_lib_all.index) // batchsize
    # valid_steps = (len(df_supertest.index) // batchsize)

    ## Laptop testmode
    if socket.gethostname().lower() in 'rs-e5550'.lower(): train_steps = 2; valid_steps=2; tensorboard=True


    iterate = K.function([input_data, K.learning_phase()], [y_pred])
    test_cb = TestCallback(iterate, validdata)

    cp_cb = ModelCheckpoint(filepath='./checkpoints/epoch/{}_epoch_check.hdf5'.format(runtimestr), verbose=1, save_best_only=False)
    tb_cb = BlankCallback()

    if tensorboard:
        tb_cb = TensorBoard(log_dir='./tensorboard/{}/'.format(runtimestr), histogram_freq=1, write_graph=True, write_images=True)

    model.fit_generator(generator=traindata.next_batch(),
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        callbacks=[cp_cb, tb_cb, test_cb, traindata, validdata],  ##create custom callback to handle stop for valid
                        validation_data=validdata.next_batch(),
                        validation_steps=1,
                        initial_epoch=0
                        )

    ## 5. final test - move this to run-test
    # model.predict_generator(testdata.next_batch(), 8, workers=1, verbose=1)

    ## save final version of model
    save_model(model, name="./checkpoints/fin/{}_ds_ctc_FIN".format(runtimestr))



#######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--loadcheckpoint', type=bool, default=False,
                       help='If true, load for the last checkpoint at the default path we find')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=16,
                       help='batch_size used to train the model')
    parser.add_argument('--tensorboard', type=bool, default=True,
                       help='batch_size used to train the model')
    parser.add_argument('--deepspeech', type=int, default=1,
                       help='choose between deepspeech versions (when training not loading) '
                            '--deepspeech=1 uses fully connected layers with simplernn'
                            '--deepspeech=2 uses fully connected with GRU')

    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.sortagrad, args.loadcheckpoint, args.epochs, args.batchsize, args.tensorboard, runtime, args.deepspeech)


###
#todo this is not a ready output
#testdata.export_test_mfcc()


