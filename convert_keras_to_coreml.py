from __future__ import unicode_literals

import argparse

from model import *
from utils import *

import keras
import coremltools

#####################################################

#### MAIN



def main(checkpointpath):

    ## hack required for clipped relu
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})


    ## Load model from checkpoint path
    loaded_model = load_model_checkpoint(checkpointpath)

    ## Try to convert assume newly trained and will will fail with CTC lambda
    print("Try convert with CoreML ")
    try:
        coreml_model = coremltools.converters.keras.convert(loaded_model)

    except Exception as e:
        print("Conversion failed - trying to rebuild without lambda")
        print(e)


        ## Rebuild function without CTC lambda and transfer weights
        model, input_data, y_pred = build_ds1_simple_rnn_no_ctc_and_xfer_weights(loaded_model=loaded_model,
                                                             fc_size=2048,
                                                             rnn_size=512,
                                                             mfcc_features=26,
                                                             dropout=[0.0, 0.0, 0.0],
                                                             num_classes=29)

        # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        # model.compile(loss='', optimizer=sgd)
        print(model.summary(line_length=80))

        print("Retry converting new model")
        coreml_model = coremltools.converters.keras.convert(model)



    # Set model metadata
    coreml_model.author = 'Rob Smith'
    coreml_model.license = 'BSD'
    coreml_model.short_description = 'Performs keras ds '
    coreml_model.input_description['input1'] = 'Audio input'
    coreml_model.output_description['output1'] = 'Audio transcription'

    # SAVE CoreML
    coreml_model.save('kds.mlmodel')

    ##Export the trimmed model (without CTC) to test that it works on python
    save_trimmed_model(model, name='./checkpoints/trimmed/TRIMMED_ds_model')
    print("Completed")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ##defaults to the finished checkpoint
    parser.add_argument('--checkpointpath', type=str, #default="./checkpoints/fin/"
                       #"DS1_2017-08-11_14-06_ds_ctc_FIN_loss151",
                        default="./checkpoints/DS1_2017-08-11_13-47_epoch_check",
                       help='checkpoint path to look in')

    args = parser.parse_args()

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.checkpointpath)

