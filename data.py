import fnmatch
import os
import pandas as pd
import char_map
from utils import text_to_int_sequence
import subprocess

#######################################################

def read_text(full_wav):
    # need to remove _rif.wav (8chars) then add .TXT
    trans_file = full_wav[:-8] + ".TXT"
    with open(trans_file, "r") as f:
        for line in f:
            split = line.split()
            start = split[0]
            end = split[1]
            t_list = split[2:]
            trans = ""
        # insert cleaned word (lowercase plus removed bad punct)
        for i, w in enumerate(t_list):
            if(i==0):
                trans = trans + clean(w)
            else:
                trans = trans + ' ' + clean(w)

        # trans = trans + '\n' ## temp output for wordlist file

    return start, end, trans

# token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
def clean(word):
    ## LC ALL & strip fullstop, comma and semi-colon which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new



def get_all_wavs_in_path(target, sortagrad=True):
    '''

    Builds a list of the wavs, transcriptions and finish times (for sorting) of a directory

    :param target: directory to search for wavs
    :param sortagrad: sort all/training dataframes
    :return: dataproperties dict and 4 dataframes (all/train/valid/test)

    '''

    train_list_wavs, train_list_trans, train_list_fin = [], [], []
    valid_list_wavs, valid_list_trans, valid_list_fin = [], [], []
    test_list_wavs, test_list_trans, test_list_fin = [], [], []

    file_count = 0
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.wav"):

            full_wav = os.path.join(root, filename)
            _, end, trans = read_text(full_wav)

            if 'train' in full_wav.lower():
                train_list_wavs.append(full_wav)
                train_list_trans.append(trans)
                train_list_fin.append(end)

            elif 'test' in full_wav.lower():
                ##split 50/50 into validation and test (note not random)
                if file_count % 2 == 0:
                    test_list_wavs.append(full_wav)
                    test_list_trans.append(trans)
                    test_list_fin.append(end)
                else:
                    valid_list_wavs.append(full_wav)
                    valid_list_trans.append(trans)
                    valid_list_fin.append(end)
            else:
                raise IOError

            file_count = file_count + 1


    a = {'wav_filename': train_list_wavs,
         'wav_filesize': train_list_fin,
         'transcript': train_list_trans}

    b = {'wav_filename': valid_list_wavs,
         'wav_filesize': valid_list_fin,
         'transcript': valid_list_trans}

    c = {'wav_filename': test_list_wavs,
         'wav_filesize': test_list_fin,
         'transcript': test_list_trans}

    al = {'wav_filename': train_list_wavs+valid_list_wavs+test_list_wavs,
         'wav_filesize': train_list_fin+valid_list_fin+test_list_fin,
         'transcript': train_list_trans+valid_list_trans+test_list_trans}


    df_all = pd.DataFrame(al, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_train = pd.DataFrame(a, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_valid = pd.DataFrame(b, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    df_test = pd.DataFrame(c, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)

    # if sortagrad is enabled sort the data for the first epoch (training only)
    if sortagrad:
        df_all = df_all.sort_values(by='wav_filesize', ascending=True)
        df_train = df_train.sort_values(by='wav_filesize', ascending=True)


    comb = train_list_trans + test_list_trans + valid_list_trans
    print("All combined:", len(comb))
    print("Train/Test/Valid:",len(train_list_wavs), len(test_list_wavs), len(valid_list_wavs))
    # 6300 TIMIT
    # (4620, 840, 840) TIMIT

    ## SIZE CHECKS
    max_intseq_length = get_max_intseq(comb)
    num_classes = get_number_of_char_classes()

    print("max_intseq_length:", max_intseq_length)
    print("numclasses:", num_classes)

    # VOCAB CHECKS
    all_words, max_trans_charlength = get_words(comb)
    print("max_trans_charlength:", max_trans_charlength)
    # ('max_trans_charlength:', 80)

    all_vocab = set(all_words)
    print("Words:", len(all_words))
    print("Vocab:", len(all_vocab))

    dataproperties = {
        'target': target,
        'num_classes': num_classes,
        'all_words': all_words,
        'all_vocab': all_vocab,
        'max_trans_charlength': max_trans_charlength,
        'max_intseq_length': max_intseq_length
    }

    # create a lm from all words
    # try:
    # print("Trying to build 4-gram LM from all data")
    #
    # with open('./lm/word_list.txt', 'w') as f:
    #     f.write(' '.join(comb)) #todo handle newlines
    #
    # ps = subprocess.Popen(('cat','./lm/word_list.txt'), stdout=subprocess.PIPE)
    # args = "-o 4 "
    # output = subprocess.check_call(["/home/rob/Dropbox/UCL/DIS/kenlm/kenlm/bin/lmplz", args], stdin=ps.stdout) #todo
    # ps.wait()

    # except Exception as e:
    #     print("Couldn't run language model build command")
    #     print(e)
    #     raise(e)

    return dataproperties, df_all, df_train, df_valid, df_test




def check_all_wavs_and_trans_from_csvs(csvs, timit, sortagrad=True):

    #passed in df_frame
    df_all = timit

    for csv in csvs.split(','):
        print("Reading csv:",csv)
        df_new = pd.read_csv(csv, sep=',')
        df_all = df_all.append(df_new)

    print("Finished reading in data")

    # df_all['transcript'].to_csv("./lm/df_all_libri_timit_word_list.csv", sep=',', header=False, index=False)  # reorder + out

    listcomb = df_all['transcript'].tolist()
    print("Total number of files:", len(listcomb))

    print("removing any sentences that are too big- tweetsize")
    df_final = df_all[df_all['transcript'].map(len) <= 140]

    listcomb = df_final['transcript'].tolist()
    print("Total number of files (after reduction):", len(listcomb))

    comb = []

    for t in listcomb:
        comb.append(' '.join(t.split()))

    # print("Train/Test/Valid:",len(train_list_wavs), len(test_list_wavs), len(valid_list_wavs))
    # 6300 TIMIT
    # (4620, 840, 840) TIMIT



    ## SIZE CHECKS
    max_intseq_length = get_max_intseq(comb)
    num_classes = get_number_of_char_classes()

    print("max_intseq_length:", max_intseq_length)
    print("numclasses:", num_classes)

    # VOCAB CHECKS
    all_words, max_trans_charlength = get_words(comb)
    print("max_trans_charlength:", max_trans_charlength)
    # ('max_trans_charlength:', 80)

    ## TODO could readd the mfcc checks for safety
    # ('max_mfcc_len:', 778, 'at comb index:', 541)

    all_vocab = set(all_words)
    print("Words:", len(all_words))
    print("Vocab:", len(all_vocab))

    dataproperties = {
        'target': "timit+librispeech",
        'num_classes': num_classes,
        'all_words': all_words,
        'all_vocab': all_vocab,
        'max_trans_charlength': max_trans_charlength,
        'max_intseq_length': max_intseq_length
    }

    if sortagrad:
        df_final = df_final.sort_values(by='wav_filesize', ascending=True)

    return dataproperties, df_final


def get_data_from_pandas_files(target, sortagrad=True):
    '''Different approach to loading in data. Assume that pandas frame already exists with data in form
        path, size, transcript
        this is best approach for loading in moz deepspeech processed files.
    '''

    ## read in file

    ## sortagrad

    ## run checks (might need to adjust nn or drop any too large)

    pass
    return





##DATA CHECKS RUN ALL OF THESE

def get_words(comb):
    max_trans_charlength = 0
    all_words = []

    for count, sent in enumerate(comb):
        # count length
        if len(sent) > max_trans_charlength:
            max_trans_charlength = len(sent)
        # build vocab
        for w in sent.split():
            all_words.append(clean(w))

    return all_words, max_trans_charlength

def get_max_intseq(comb):
    max_intseq_length = 0
    for x in comb:
        try:
            y = text_to_int_sequence(x)
            if len(y) > max_intseq_length:
                max_intseq_length = len(y)
        except:
            print("error at:", x)
    return max_intseq_length

def get_number_of_char_classes():
    ## TODO would be better to check with dataset (once cleaned)
    num_classes = len(char_map.char_map)+1 ##need +1 for ctc null char +1 pad
    return num_classes
