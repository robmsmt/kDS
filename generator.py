import keras.callbacks
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import python_speech_features as p
import scipy.io.wavfile as wav
import editdistance  # todo use tf edit dist


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

    def wer(self):
        #todo
        pass

    def validate_epoch_end(self):
        mean_norm_ed = 0.0
        mean_ed = 0.0
        self.validdata.cur_index = 0  # reset index

        chunks = len(self.validdata.wavpath) // self.validdata.batch_size

        #call next batch until all data

        for c in range(0, chunks):

            print(self.validdata.cur_index)
            word_batch = next(self.validdata_next_val)[0]
            #num_proc = batch_size #min of batchsize OR num_left

            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:self.batch_size],
                                       word_batch['source_str'][0:self.batch_size],
                                       self.batch_size)

            #todo LM can go here
            #todo WER more used metric

            for j in range(0, self.batch_size):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j]) ## todo test edit distance with strings
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
                if(j % 8 == 0):
                    print("\n{}.Truth:{}\n{}.Trans:{}".format(str(j), word_batch['source_str'][j], str(j), decoded_res[j]))

            mean_norm_ed = mean_norm_ed
            mean_ed = mean_ed

            print('\nOut of %d batch samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
                  % (1, mean_ed, mean_norm_ed))


    def on_epoch_end(self, epoch, logs=None):

        self.validate_epoch_end()
        #word_batch = next(self.validdata_next_val)[0]
        #result = decode_batch(self.test_func, word_batch['the_input'][0])
        #print("Truth: {} \nTranscribed: {}".format(word_batch['source_str'], result[0]))



