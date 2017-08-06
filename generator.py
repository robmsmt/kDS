import keras.callbacks
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import python_speech_features as p
import scipy.io.wavfile as wav
import editdistance  # todo use tf edit dist

import kenlm
import re
from heapq import heapify

from keras.preprocessing.sequence import pad_sequences
from keras import callbacks

from utils import decode_batch, text_to_int_sequence



def get_intseq(trans,max_intseq_length=80):
    # PAD
    t = text_to_int_sequence(trans)
    while (len(t) < max_intseq_length):
        t.append(0)  # replace with a space char to pad
    # print(t)
    return t

def get_mfcc(filename):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=778, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}

def get_xsize(val):
    return val.shape[0]


def get_ylen(val):
    return len(val)



class BaseGenerator(callbacks.Callback):
    def __init__(self, dataframe, dataproperties, batch_size=16, max_intseq_length=80):
        self.df = dataframe
        self.dataproperties = dataproperties
        self.wavpath = self.df['wavs'].tolist()
        self.transcript = self.df['trans'].tolist()
        self.finish = self.df['fin'].tolist()
        self.start = np.zeros(len(self.finish))
        self.length = self.finish

        self.batch_size = batch_size
        self.cur_index = 0
        self.max_intseq_length = 80  # todo extract this from dataproperties for all datasets

        self.feats_std = 0
        self.feats_mean = 0

        self.set_of_all_int_outputs_used = None

    def normalise(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def get_batch(self, idx):

        batch_x = self.wavpath[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_trans = self.transcript[idx * self.batch_size:(idx + 1) * self.batch_size]

        try:
            assert (len(batch_x) == self.batch_size)
            assert (len(batch_y_trans) == self.batch_size)
        except Exception as e:
            print(e)
            print(batch_x)
            print(batch_y_trans)

        # 1. X_data (the MFCC's for the batch)
        X_data = np.array([get_mfcc(file_name) for file_name in batch_x])
        assert (X_data.shape == (self.batch_size, 778, 26))

        # TODO batch-normalisation - later as will affect training
        # features = [self.featurize(a) for a in batch_x]
        # for i in range(self.batch_size):
        #     feat = features[i]
        #     feat = self.normalize(feat)

        # 2. labels (made numerical)
        labels = np.array([get_intseq(l, self.max_intseq_length) for l in batch_y_trans])
        assert (labels.shape == (self.batch_size, self.max_intseq_length))

        # 3. input_length (required for CTC loss)
        input_length = np.array([get_xsize(mfcc) for mfcc in X_data])
        assert (input_length.shape == (self.batch_size,))

        # 4. label_length (required for CTC loss)
        label_length = np.array([get_ylen(y) for y in labels])
        assert (label_length.shape == (self.batch_size,))

        # 5. source_str (used for human readable output on callback)
        source_str = np.array([l for l in batch_y_trans])

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str
        }

        outputs = {'ctc': np.zeros([self.batch_size])}

        return (inputs, outputs)

    def next_batch(self):
        while 1:
            assert (self.batch_size <= len(self.wavpath))
            if (self.cur_index + 1) * self.batch_size >= len(self.wavpath) - self.batch_size:
                self.cur_index = 0
                self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                                     self.transcript,
                                                                     self.finish)
            ret = self.get_batch(self.cur_index)
            self.cur_index += 1

            yield ret

    def get_normalise(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        # k_samples = min(k_samples, len(self.train_audio_paths))
        # samples = self.rng.sample(self.train_audio_paths, k_samples)
        # feats = [self.featurize(s) for s in samples]
        # feats = np.vstack(feats)
        # self.feats_mean = np.mean(feats, axis=0)
        # self.feats_std = np.std(feats, axis=0)
        pass

    def on_epoch_end(self, epoch, logs={}):
        print("EPOCH END - shuffling data")
        self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                             self.transcript,
                                                             self.finish)

#
# class TestValidGenerator(BaseGenerator):
#     def __init__(self):
#         super(BaseGenerator, self).__init__()
#
#     '''
#     Extends the BaseGenerator class as specific functions will be required for test/valid data
#
#     '''
#
#     def export_test_mfcc(self):
#         # this is used to export data e.g. into iOS
#
#         testset = next(self.next_batch())[0]
#         mfcc = testset['the_input'][0:self.batch_size] ## export all mfcc's in batch #26 x 29 ?
#         words = testset['source_str'][0:self.batch_size]
#         labels = testset['the_labels'][0:self.batch_size]
#
#         print("exporting:", type(mfcc))
#         print(mfcc.shape)
#         print(words.shape)
#         print(labels.shape)
#
#         # we save each mfcc/words/label as it's own csv file
#         for i in range(0, mfcc.shape[0]):
#             np.savetxt('./test_mfccs/test_mfcc_{}.csv'.format(i), mfcc[i,:,:], delimiter=',')
#             np.savetxt('./test_mfccs/test_words_{}.csv'.format(i), words[i,:], delimiter=',')
#             np.savetxt('./test_mfccs/test_labels_{}.csv'.format(i), labels[i,:], delimiter=',')
#
#         return




class TestCallback(callbacks.Callback):
    def __init__(self, test_func, validdata):
        self.test_func = test_func
        self.validdata = validdata
        self.validdata_next_val = self.validdata.next_batch()
        self.batch_size = validdata.batch_size

        self.val_best_mean_ed = 0
        self.val_best_norm_mean_ed = 0

        self.lm = get_model()


    def validate_epoch_end(self):
        mean_norm_ed = 0.0
        mean_ed = 0.0
        self.validdata.cur_index = 0  # reset index

        chunks = len(self.validdata.wavpath) // self.validdata.batch_size

        #make a pass through all the validation data and assess score
        for c in range(0, chunks):

            print(self.validdata.cur_index)
            word_batch = next(self.validdata_next_val)[0]
            #num_proc = batch_size #min of batchsize OR num_left

            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:self.batch_size],
                                       word_batch['source_str'][0:self.batch_size],
                                       self.batch_size)


            for j in range(0, self.batch_size):

                decode_sent = decoded_res[j]
                corrected = correction(decode_sent)
                label = word_batch['source_str'][j]
                sample_wer = wer(label, corrected)

                print("\n{}.STruth :{}\n{}.OTrans :{}\n{}.Correct:{}".format(str(j), label,
                                                                             str(j), decode_sent,
                                                                             str(j), corrected)
                     )

                print(sample_wer)

                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j]) ## todo test edit distance with strings
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
                # if(j % 16 == 0):
                #     print("\n{}.Truth:{}\n{}.Trans:{}".format(str(j), word_batch['source_str'][j], str(j), decoded_res[j]))

            mean_norm_ed = mean_norm_ed
            mean_ed = mean_ed

            print('\nOut of %d batch samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
                  % (1, mean_ed, mean_norm_ed))


    def on_epoch_end(self, epoch, logs=None):

        self.validate_epoch_end()
        #word_batch = next(self.validdata_next_val)[0]
        #result = decode_batch(self.test_func, word_batch['the_input'][0])
        #print("Truth: {} \nTranscribed: {}".format(word_batch['source_str'], result[0]))


def wer(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    return levenshtein(original, result) / float(len(original))

def wers(originals, results):
    count = len(originals)
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)

# The following code is from: http://hetland.org/coding/python/levenshtein.py

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


# Define beam with for alt sentence search
BEAM_WIDTH = 1024
MODEL = None


# Lazy-load language model (TED corpus, Kneser-Ney, 4-gram, 30k word LM)
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = kenlm.Model('./lm/timit-lm.klm')
    return MODEL


def words(text):
    "List of words in text."
    return re.findall(r'\w+', text.lower())


# Load known word set
with open('./lm/word_list.txt') as f:
    WORDS = set(words(f.read()))


def log_probability(sentence):
    "Log base 10 probability of `sentence`, a list of words"
    return get_model().score(' '.join(sentence), bos=False, eos=False)


def correction(sentence):
    "Most probable spelling correction for sentence."
    layer = [(0, [])]
    for word in words(sentence):
        layer = [(-log_probability(node + [cword]), node + [cword]) for cword in candidate_words(word) for
                 priority, node in layer]
        heapify(layer)
        layer = layer[:BEAM_WIDTH]
    return ' '.join(layer[0][1])


def candidate_words(word):
    "Generate possible spelling corrections for word."
    return (known_words([word]) or known_words(edits1(word)) or known_words(edits2(word)) or [word])


def known_words(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

