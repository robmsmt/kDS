
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import python_speech_features as p
import scipy.io.wavfile as wav
import editdistance  # todo use tf edit dist

import kenlm
import re
from heapq import heapify
import socket

from keras.preprocessing.sequence import pad_sequences
from keras import callbacks
from numpy.lib.stride_tricks import as_strided

from utils import decode_batch, text_to_int_sequence, save_model

import soundfile

def get_maxseq_len(trans):
    # PAD
    t = text_to_int_sequence(trans)
    return len(t)

def get_intseq(trans, max_intseq_length=80):
    # PAD
    t = text_to_int_sequence(trans)
    while (len(t) < max_intseq_length):
        t.append(27)  # replace with a space char to pad
    # print(t)
    return t

def get_max_time(filename):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    return r.shape[0]  #

def get_max_specto_time(filename):
    r = spectrogram_from_file(filename)
    # print(r.shape)
    return r.shape[0]  #

def make_specto_shape(filename,padlen=778):
    r = spectrogram_from_file(filename)
    t = np.transpose(r)  # 2D array ->  spec x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T

    return X # MAXtimesamples x specto {max x 161}

def make_mfcc_shape(filename,padlen=778):
    fs, audio = wav.read(filename)
    r = p.mfcc(audio, samplerate=fs, numcep=26)  # 2D array -> timesamples x mfcc_features
    t = np.transpose(r)  # 2D array ->  mfcc_features x timesamples
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T
    return X  # 2D array -> MAXtimesamples x mfcc_features {778 x 26}

def get_xsize(val):
    return val.shape[0]


class BaseGenerator(object):
    def __init__(self, dataframe, dataproperties, training, batch_size=16, spectogram=False):
        self.training_data = training
        self.spectogram = spectogram
        self.df = dataframe
        self.dataproperties = dataproperties
        #['wav_filesize','transcript','wav_filename']
        self.wavpath = self.df['wav_filename'].tolist()
        self.transcript = self.df['transcript'].tolist()
        self.finish = self.df['wav_filesize'].tolist()
        self.start = np.zeros(len(self.finish))
        self.length = self.finish

        self.batch_size = batch_size
        self.cur_index = 0

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
        if(self.spectogram):
            # 0. get the maximum time length of the batch
            x_val = [get_max_specto_time(file_name) for file_name in batch_x]
            max_val = max(x_val)
            # print("Max batch time value is:", max_val)

            X_data = np.array([make_specto_shape(file_name, padlen=max_val) for file_name in batch_x])
            assert (X_data.shape == (self.batch_size, max_val, 161))

        else:
            # 0. get the maximum time length of the batch
            x_val = [get_max_time(file_name) for file_name in batch_x]
            max_val = max(x_val)
            # print("Max batch time value is:", max_val)

            X_data = np.array([make_mfcc_shape(file_name, padlen=max_val) for file_name in batch_x])
            assert (X_data.shape == (self.batch_size, max_val, 26))
        # print("1. X_data shape:", X_data.shape)
        # print("1. X_data:", X_data)


        # 2. labels (made numerical)
        # get max label length
        y_val = [get_maxseq_len(l) for l in batch_y_trans]
        # print(y_val)
        max_y = max(y_val)
        # print(max_y)
        labels = np.array([get_intseq(l, max_intseq_length=max_y) for l in batch_y_trans])
        # print("2. labels shape:", labels.shape)
        # print("2. labels values=", labels)
        assert(labels.shape == (self.batch_size, max_y))

        # 3. input_length (required for CTC loss)
        # this is the time dimension of CTC (batch x time x mfcc)
        #input_length = np.array([get_xsize(mfcc) for mfcc in X_data])
        input_length = np.array(x_val)
        # print("3. input_length shape:", input_length.shape)
        # print("3. input_length =", input_length)
        assert(input_length.shape == (self.batch_size,))

        # 4. label_length (required for CTC loss)
        # this is the length of the number of label of a sequence
        #label_length = np.array([len(l) for l in labels])
        label_length = np.array(y_val)
        # print("4. label_length shape:", label_length.shape)
        # print("4. label_length =", label_length)
        assert(label_length.shape == (self.batch_size,))

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
                print("EVERYDAY I AM SHUFFLING SHUFFLING")
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

    # def on_epoch_end(self, epoch, logs={}):
    #     if(self.training_data) and False:
    #         print("EPOCH END - shuffling data")
    #         self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
    #                                                          self.transcript,
    #                                                          self.finish)


    def shuffle_data(self):
        self.wavpath, self.transcript, self.finish = shuffle(self.wavpath,
                                                             self.transcript,
                                                             self.finish)
        return


    def export_test_mfcc(self):
        # this is used to export data e.g. into iOS

        testset = next(self.next_batch())[0]
        mfcc = testset['the_input'][0:self.batch_size] ## export all mfcc's in batch #26 x 29 ?
        words = testset['source_str'][0:self.batch_size]
        labels = testset['the_labels'][0:self.batch_size]

        print("exporting:", type(mfcc))
        print(mfcc.shape)
        print(words.shape)
        print(labels.shape)

        # we save each mfcc/words/label as it's own csv file
        for i in range(0, mfcc.shape[0]):
            np.savetxt('./test_mfccs/test_mfcc_{}.csv'.format(i), mfcc[i,:,:], delimiter=',')

        print(words)
        print(labels)

        return


class BlankCallback(callbacks.Callback):
    #used incase tensorboard not required
    pass



class TestCallback(callbacks.Callback):
    def __init__(self, test_func, validdata, traindata, model, runtimestr, decccc):
        self.test_func = test_func
        self.test_decccc = decccc
        self.validdata = validdata
        self.traindata = traindata
        self.validdata_next_val = self.validdata.next_batch()
        self.batch_size = validdata.batch_size

        self.val_best_mean_ed = 0
        self.val_best_norm_mean_ed = 0

        self.lm = get_model()

        self.model = model
        self.runtimestr = runtimestr

        self.mean_wer_log = []
        self.mean_ler_log = []
        self.norm_mean_ler_log = []


    def validate_epoch_end(self, verbose=0):

        originals = []
        results = []
        count = 0
        self.validdata.cur_index = 0  # reset index

        allvalid = len(self.validdata.wavpath) // self.validdata.batch_size

        if socket.gethostname().lower() in 'rs-e5550'.lower(): allvalid = 2


        #make a pass through all the validation data and assess score
        for c in range(0, allvalid):

            word_batch = next(self.validdata_next_val)[0]
            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:self.batch_size],
                                       word_batch['source_str'][0:self.batch_size],
                                       self.batch_size)


            for j in range(0, self.batch_size):
                # print(c,j)
                count += 1
                decode_sent = decoded_res[j]
                corrected = correction(decode_sent)
                label = word_batch['source_str'][j]

                if verbose:
                    cor_wer = wer(label, corrected)
                    dec_wer = wer(label, decode_sent)

                    if(dec_wer < 0.5 or cor_wer < 0.5):
                        print("\n{}.GroundTruth:{}\n{}.Transcribed:{}\n{}.LMCorrected:{}".format(str(j), label,
                                                                                     str(j), decode_sent,
                                                                                     str(j), corrected))

                    # print("Sample Decoded WER:{}, Corrected LM WER:{}".format(dec_wer, cor_wer))

                originals.append(label)
                results.append(corrected)

        print("########################################################")
        print("Completed Validation Test: WER & LER results")
        rates, mean = wers(originals, results)
        # print("WER rates     :", rates)
        lrates, lmean, norm_lrates, norm_lmean = lers(originals, results)
        # print("LER rates     :", lrates)
        # print("LER norm rates:", norm_lrates)
        # print("########################################################")
        print("Test WER average is   :", mean)
        print("Test LER average is   :", lmean)
        print("Test normalised LER is:", norm_lmean)
        print("########################################################")
        # print("(note both WER and LER use LanguageModel not raw output)")

        self.mean_wer_log.append(mean)
        self.mean_ler_log.append(lmean)
        self.norm_mean_ler_log.append(norm_lmean)

        #delete all values?
        # del originals, results, count, allvalid
        # del word_batch, decoded_res
        # del decode_sent,



    def on_epoch_end(self, epoch, logs=None):

        self.validate_epoch_end(verbose=1)
        save_model(self.model, name="./checkpoints/epoch/{}_epoch_check".format(self.runtimestr))
        self.traindata.shuffle_data()
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

def lers(originals, results):
    count = len(originals)
    rates = []
    norm_rates = []

    mean = 0.0
    norm_mean = 0.0


    assert count == len(results)
    for i in range(count):
        rate = levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean/float(count))


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
        #MODEL = kenlm.Model('./lm/timit-lm.klm')
        MODEL = kenlm.Model('./lm/libri-timit-lm.klm')
    return MODEL


def words(text):
    "List of words in text."
    return re.findall(r'\w+', text.lower())


# Load known word set
with open('./lm/df_all_libri_timit_word_list.csv') as f:
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


##############################################################################

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs



def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))

def featurise(audio_clip):
    """ For a given audio clip, calculate the log of its Fourier Transform
    Params:
        audio_clip(str): Path to the audio clip
    """

    step = 10
    window = 20
    max_freq = 8000

    return spectrogram_from_file(
        audio_clip, step=step, window=window,
        max_freq=max_freq)

