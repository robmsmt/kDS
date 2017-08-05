from __future__ import unicode_literals

import argparse

from model import *
from utils import *

import keras
import coremltools

#####################################################

#### MAIN


def main(checkpointpath):


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
                                                             max_mfcclength_audio=778,
                                                             dropout=[0.2, 0.5, 0.3],
                                                             num_classes=30)

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
    save_model(model, name='./checkpoints/TRIMMED_ds_model')
    print("Completed")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ##defaults to the finished checkpoint
    parser.add_argument('--checkpointpath', type=str, default="./checkpoints/ds_ctc_FIN",
                       help='If true, we sort utterances by their length in the first epoch')

    args = parser.parse_args()

    assert(keras.__version__ == "2.0.4") ## CoreML is super strict

    main(args.checkpointpath)

