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
from keras.optimizers import Adam


#######################################################

# Prevent pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args, runtime):
    '''
    There are 5 simple steps to this program
    '''

    runtimestr = "DS"+str(args.deepspeech)+"_"+runtime

    ## 1. get path and data
    timit_path = get_timit_data_path()
    timit_dataproperties, df_all, df_train, df_valid, df_test = get_all_wavs_in_path(timit_path, sortagrad=args.sortagrad)

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

    lib_dataproperties, df_lib_all = check_all_wavs_and_trans_from_csvs(csvs, df_train)


    ## 2. init data generators
    traindata = BaseGenerator(dataframe=df_lib_all, dataproperties=lib_dataproperties, training=True, batch_size=args.batchsize)
    validdata = BaseGenerator(dataframe=df_supertest, dataproperties=timit_dataproperties, training=False, batch_size=args.batchsize)


    ## 3. Load existing or create new model
    if args.loadcheckpointpath:
        # load existing -todo test this

        assert(os.path.isfile(args.loadcheckpointpath))

        #todo think this is needed for callback
        _, input_data, y_pred = ds1(fc_size=2048,
                                                         rnn_size=512,
                                                         mfcc_features=26,
                                                         num_classes=30) # required for callback todo test
        print(input_data, y_pred)

        model = load_model_checkpoint(custom_objects={'ctc': ctc}, path=args.loadcheckpointpath)
        input_data = model.inputs[0]
        y_pred = model.outputs[0]

        print(input_data, y_pred)

    else:
        # new model
        if(args.deepspeech==1):
            print('DS{}'.format(args.deepspeech))
            model, input_data, y_pred, input_length = ds1(fc_size=2048, rnn_size=512,mfcc_features=26,num_classes=29)
            opt = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        elif(args.deepspeech==2):
            print('DS{}'.format(args.deepspeech))
            model, input_data, y_pred, input_length = ds2_gru_model(input_dim=26, output_dim=29, nodes=1024,
                                                      initialization='glorot_uniform')
            opt = Adam(lr=0.001, clipnorm=5)

        elif(args.deepspeech==3):
            print('DS{}'.format(args.deepspeech))
            model, input_data, y_pred, input_length = ds1(fc_size=2048, rnn_size=512, mfcc_features=26, num_classes=29)
            opt = Adam(lr=0.001, clipnorm=5)


        # Compile with dummy loss
        model.compile(loss=ctc, optimizer=opt)
        print(model.summary(line_length=80))

    ## 4. train
    train_steps = len(df_lib_all.index) // args.batchsize
    # valid_steps = (len(df_supertest.index) // batchsize)

    ## Laptop testmode
    if socket.gethostname().lower() in 'rs-e5550'.lower(): train_steps = 4; args.tensorboard=False; args.epochs=4


    iterate = K.function([input_data, K.learning_phase()], [y_pred])
    # decode = K.function([y_pred, input_length], [dec])
    decode = None # temp

    test_cb = TestCallback(iterate, validdata, model, runtimestr, decode)
    tb_cb = BlankCallback()

    if args.tensorboard:
        tb_cb = TensorBoard(log_dir='./tensorboard/{}/'.format(runtimestr), histogram_freq=1, write_graph=True, write_images=True)

    model.fit_generator(generator=traindata.next_batch(),
                        steps_per_epoch=train_steps,
                        epochs=args.epochs,
                        callbacks=[tb_cb, test_cb, traindata, validdata],  ##create custom callback to handle stop for valid
                        # validation_data=validdata.next_batch(),
                        # validation_steps=1,
                        initial_epoch=0
                        )

    ## These are the most important metrics
    print("Mean WER   :", test_cb.mean_wer_log)
    print("Mean LER   :", test_cb.mean_ler_log)
    print("NormMeanLER:",test_cb.norm_mean_ler_log)

    ## 5. final test - move this to run-test
    res = model.evaluate_generator(validdata.next_batch(), 8, max_q_size=10, workers=1)
    print(res)


    ## save final version of model
    save_model(model, name="./checkpoints/fin/{}_ds_ctc_FIN_loss{}".format(runtimestr,int(res)))

    ###
    # todo this is not a ready output
    # validdata.export_test_mfcc()

#######################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--loadcheckpointpath', type=str, default='', # ./checkpoints/fin/ds_ctc_model_epoch_end.hdf5',
                       help='If value set, load the checkpoint json '
                            'weights assumed as same name _weights'
                            ' e.g. --loadcheckpointpath ./checkpoints/'
                            'TRIMMED_ds_ctc_model.json ')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=16,
                       help='batch_size used to train the model')
    parser.add_argument('--tensorboard', type=bool, default=True,
                       help='batch_size used to train the model')
    parser.add_argument('--deepspeech', type=int, default=1,
                       help='choose between deepspeech versions (when training not loading) '
                            '--deepspeech=1 uses fully connected layers with simplernn'
                            '--deepspeech=2 uses fully connected with GRU'
                            '--deepspeech=3 is custom model')
    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args, runtime)





