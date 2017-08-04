from char_map import char_map, index_map
import itertools
import numpy as np
import socket
from keras.models import model_from_json

import tensorflow as tf

def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

def int_to_text_sequence(seq):
    """ Use a index map and convert int to a text sequence """
    text_sequence = []
    for c in seq:
        if c == 29: #ctc char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):

        ## todo replace with greedy/beam search

        # # Beam search decode the batch
        # decoded, _ = tf.nn.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)
        #
        # # Compute the edit (Levenshtein) distance
        # distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        try:
            outStr = int_to_text_sequence(out_best)
        except Exception as e:
            print("error:", e)
            outStr = "DECODE ERROR:"+str(out_best)

            # raise("DECODE ERROR2")

        ret.append(''.join(outStr))

    return ret


def save_model(model, name="./checkpoints/ds_ctc_model"):
    jsonfilename = str(name) + ".json"
    weightsfilename = str(name) + "_weights.h5"

    # # serialize model to JSON
    with open(jsonfilename, "w") as json_file:
        json_file.write(model.to_json())

    # # serialize weights to HDF5
    model.save_weights(weightsfilename)

    return

def load_model_checkpoint(path="./checkpoints/ds_ctc_model", summary=True):
    jsonfilename = path+".json"
    weightsfilename = path+"_weights.h5"

    json_file = open(jsonfilename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights(weightsfilename)

    if(summary):
        loaded_model.summary()

    return loaded_model



def get_data_path():
    ##TODO - make this work for all datasets
    ## Use hostname to detect my laptop OR else it's cluster
    hostname = socket.gethostname().lower()
    if hostname in ('rs-e5550').lower():
        datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LDC/timit/"
    else:
        datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/timit/"
    target = datapath + "TIMIT/"
    return target
#
#>>> from utils import int_to_text_sequence
#>>> a = [2,22,10,11,21,2,13,11,6,1,21,2,8,20,17]
#>>> b = int_to_text_sequence(a)

