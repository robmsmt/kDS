from char_map import char_map, index_map

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
        for t in t_list:
            trans = trans + ' ' + clean(t)

    return start, end, trans


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
        ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence
