from char_map import char_map, index_map
import itertools
import numpy as np
import socket
from keras.models import model_from_json

# import kenlm
# import re
# from heapq import heapify

# import tensorflow as tf
# from tensorflow.python.ops import array_ops

# Define beam with for alt sentence search
# BEAM_WIDTH = 1024
# MODEL = None


# # Lazy-load language model (TED corpus, Kneser-Ney, 4-gram, 30k word LM)
# def get_model():
#     global MODEL
#     if MODEL is None:
#         MODEL = kenlm.Model('../../DeepSpeech/data/lm/lm.binary')
#     return MODEL
#
# def words(text):
#     "List of words in text."
#     return re.findall(r'\w+', text.lower())
#
# # Load known word set
# with open('../../DeepSpeech/data/spell/words.txt') as f:
#     WORDS = set(words(f.read()))



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
        if c == 29 or c == 0: #ctc/pad char
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence

def decode_batch(test_func, word_batch, source_str, batch_size):

    ret = []
    output = test_func([word_batch])[0] #16x778x30 = batch x time x classes

    # seq_len = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
    # ## needs to be
    # logits = tf.transpose(output, (1, 0, 2))
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # batch_y = ctc_label_dense_to_sparse(source_str, seq_len, len(seq_len))
    # # Inaccuracy: label error rate
    # ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
    #                                       batch_y))
    # print(ler)

    for j in range(batch_size): # 0:batch_size
        out_best1 = list(np.argmax(output[j, :], axis=1))
        out_star = [k for k, g in itertools.groupby(out_best1)]
        try:
            outStr = int_to_text_sequence(out_star)
        except Exception as e:
            print("error:", e)
            outStr = "DECODE ERROR:"+str(out_star)
            raise("DECODE ERROR2")

        ret.append(''.join(outStr))

    ## LM

    # corrected = correction(' '.join(ret))
    #
    # print(ret)
    # print(corrected)

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

# def beam():
#     ## todo replace with greedy/beam search
#     # # Beam search decode the batch
#     tempoutput = array_ops.transpose(output, [1, 0, 2])
#     batch_seq_len = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
#     decoded, logprob = tf.nn.ctc_beam_search_decoder(tempoutput, batch_seq_len, merge_repeated=False)
#     batch_y = ctc_label_dense_to_sparse(source_str, batch_seq_len, len(batch_seq_len))
#
#     # # Compute the edit (Levenshtein) distance
#     distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)
#     mean_edit_distance = tf.reduce_mean(distance)

# def log_probability(sentence):
#     "Log base 10 probability of `sentence`, a list of words"
#     return get_model().score(' '.join(sentence), bos = False, eos = False)
#
# def correction(sentence):
#     "Most probable spelling correction for sentence."
#     layer = [(0,[])]
#     for word in words(sentence):
#         layer = [(-log_probability(node + [cword]), node + [cword]) for cword in candidate_words(word) for priority, node in layer]
#         heapify(layer)
#         layer = layer[:BEAM_WIDTH]
#     return ' '.join(layer[0][1])
#
# def candidate_words(word):
#     "Generate possible spelling corrections for word."
#     return (known_words([word]) or known_words(edits1(word)) or known_words(edits2(word)) or [word])
#
# def known_words(words):
#     "The subset of `words` that appear in the dictionary of WORDS."
#     return set(w for w in words if w in WORDS)
#
# def edits1(word):
#     "All edits that are one edit away from `word`."
#     letters    = 'abcdefghijklmnopqrstuvwxyz'
#     splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#     deletes    = [L + R[1:]               for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#     replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#     inserts    = [L + c + R               for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)
#
# def edits2(word):
#     "All edits that are two edits away from `word`."
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))