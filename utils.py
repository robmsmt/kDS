from char_map import char_map, index_map
import itertools
import numpy as np
import socket


from keras.models import model_from_json
# from keras.models import load_model
# from keras import backend as K
#
# import tensorflow as tf
#


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
        if c == 28: #ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence


# def trythis(inputbatch):
#
#     log_prediction_batch = tf.log(tf.transpose(inputbatch, perm=[1, 0, 2]) + 1e-8)
#     prediction_length_batch = tf.to_int32(tf.squeeze(inputbatch, axis=[1]))
#     decoded, log_prob = tf.nn.ctc_greedy_decoder(log_prediction_batch, prediction_length_batch)
#
#     return decoded,log_prob

def decode_batch(test_func, word_batch, source_str, batch_size):

    ret = []
    output = test_func([word_batch])[0] #16xTIMEx29 = batch x time x classes
    greedy = True
    merge_chars = True

    for j in range(batch_size):  # 0:batch_size

        if greedy:
            out = output[j]
            best = list(np.argmax(out, axis=1))

            if merge_chars:
                merge = [k for k,g in itertools.groupby(best)]

            else:
                raise ("not implemented no merge yet")

        else:
            pass
            raise("not implemented beam yet")

        try:
            outStr = int_to_text_sequence(merge)

        except Exception as e:
            print("error:", e)
            outStr = "DECODE ERROR:"+str(best)
            raise("DECODE ERROR2")

        ret.append(''.join(outStr))



    return ret

def test_decode(prediction, batch_size=1):
    ret = []
    for j in range(batch_size):  # 0:batch_size
        out_best1 = list(np.argmax(prediction[j, :], axis=1))
        out_star = [k for k, g in itertools.groupby(out_best1)]
        try:
            outStr = int_to_text_sequence(out_star)
        except Exception as e:
            print("error:", e)
            outStr = "DECODE ERROR:"+str(out_star)
            raise("DECODE ERROR2")

        ret.append(''.join(outStr))

    return ret


def save_trimmed_model(model, name="./checkpoints/trimmed/TRIMMED_ds_ctc_model"):

    jsonfilename = str(name) + ".json"
    weightsfilename = str(name) + "_weights.h5"

    # # serialize model to JSON
    with open(jsonfilename, "w") as json_file:
        json_file.write(model.to_json())

    # # serialize weights to HDF5
    model.save_weights(weightsfilename)

    return

def save_model(model, name):

    if name:
        jsonfilename = str(name) + ".json"
        weightsfilename = str(name) + "_weights.h5"

        # # serialize model to JSON
        with open(jsonfilename, "w") as json_file:
            json_file.write(model.to_json())

        model.save_weights(weightsfilename)

    return

def load_model_checkpoint(path, summary=True):
    jsonfilename = path+".json"
    weightsfilename = path+"_weights.h5"

    json_file = open(jsonfilename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into loaded model
    loaded_model.load_weights(weightsfilename)
    # loaded_model = load_model(path, custom_objects=custom_objects)

    if(summary):
        loaded_model.summary()

    return loaded_model

def get_timit_data_path():
    ## Use hostname to detect my laptop OR else it's cluster
    hostname = socket.gethostname().lower()
    if hostname in ('rs-e5550').lower():
        datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LDC/timit/"
    else:
        datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/timit/"
    target = datapath + "TIMIT/"
    return target

def get_librispeech_data_path():
    ## Use hostname to detect my laptop OR else it's cluster
    hostname = socket.gethostname().lower()
    if hostname in ('rs-e5550').lower():
        datapath = "/home/rob/Dropbox/UCL/DIS/Admin/LibriSpeech/"
    else:
        datapath = "/cluster/project2/darkspeech/rob/DeepSpeech/data/LibriSpeech/"
    target = datapath + ""
    return target

#>>> from utils import int_to_text_sequence
#>>> a = [2,22,10,11,21,2,13,11,6,1,21,2,8,20,17]
#>>> b = int_to_text_sequence(a)
