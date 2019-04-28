import os
import re
import argparse
import logging
import tqdm
import cv2
import numpy as np

def get_list_of_unigrams(corpus_file):
    charset = set()
    with open(corpus_file, 'r') as f:
        for word in f.readlines():
            word = word.strip()
            for char in word:
                charset.add(char)
    print(charset)
    return(charset)

def process_bbox(xx):
    xx = xx.split(' ')
    res = []
    for i in xx:
        tt = i.split(',')
        rj = []
        for j in tt:
            rj.append(int(j))
        res.append(rj)
    res = np.array(res)
    res = cv2.boundingRect(res)
    return(res)

def get_words_from_pagexml(xmlname,
        keep_punctuation=False,
        keep_capitals=False,
        keep_latins=False):
    """
    Returns a list of tuples. Each tuple corresponds to one word.
    """
    with open(xmlname, 'r') as f:
        xmldata = f.read().replace('\n', '')
    rexp = '<Word id="(.*?)">\s*<Coords points="(.*?)"/>\s*<TextEquiv>\s*<Unicode>(.+?)</Unicode>\s*</TextEquiv>\s*</Word>'
    words = re.findall(rexp, xmldata)
    #
    words_processed = []
    punctuation_mark_table = dict.fromkeys(map(ord, '\'‘&,.’:;"-()!·'), None)
    latin_min_mark_table = dict.fromkeys(map(ord, 'abcdefghijklmnopqrstuvwxyz'), None)
    latin_maj_mark_table = dict.fromkeys(map(ord, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), None)
    for x in words:
        if keep_punctuation is False:
            transcr_new = x[2].translate(punctuation_mark_table)
        else:
            transcr_new = x[2]
        transcr_new = transcr_new.replace('&quot', '')
        transcr_new = transcr_new.replace('quot', '')
        if keep_capitals is False:
            transcr_new = transcr_new.lower()
        if keep_latins is False:
            transcr_new = transcr_new.translate(latin_min_mark_table)
            transcr_new = transcr_new.translate(latin_maj_mark_table)
        #trascr_new = re.sub(r'&quot', '', transcr_new)
        points_new = process_bbox(x[1])
        #id_new = x[0]
        id_new = xmlname
        if(len(transcr_new) == 0):
            print('Warning! Found word with empty transcription (probably due to removed punctuation). Replacing with dummy character "#"')
            transcr_new = '#'
        words_processed.append( (id_new, points_new, transcr_new) )
    print('Found {} words in file {}.'.format(len(words_processed), xmlname))
    return(words_processed)

def read_sophia_xmls(keep_punctuation=False, keep_capitals=False, keep_latins=False):
    all_word_tuples = []
    all_xmls = []
    for x in range(1, 48):
        if x == 12:
            continue #Page 12 was omitted / doesn't exist
        all_xmls.append('data/_00{0:02d}.xml'.format(x))
    for x in all_xmls:
        all_word_tuples += get_words_from_pagexml(x, keep_punctuation=keep_punctuation, keep_capitals=keep_capitals, keep_latins=keep_latins)
    return all_word_tuples

if __name__=='__main__':
    logger = logging.getLogger('SophiaTrikoupi::dataset')
    logger.info('------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-mode', required=True, choices=['check','extract_lexicon','get_unigrams'], help='Execution mode.')
    parser.add_argument('--keep_punctuation',  dest='keep_punctuation', action='store_true', help='Do not remove punctuation from annotation.')
    parser.add_argument('--keep_capitals',  dest='keep_capitals', action='store_true', help='Do not force lowercase letters on annotation.')
    parser.add_argument('--keep_latins',  dest='keep_latins', action='store_true', help='Do not remove latin letters from annotation.')
    parser.set_defaults(
            keep_punctuation=False,
            keep_capitals=False,
            keep_latins=False,
    )
    args = parser.parse_args()
    all_word_tuples = read_sophia_xmls(keep_punctuation=args.keep_punctuation, keep_capitals=args.keep_capitals, keep_latins=args.keep_latins)

    if args.mode == 'check':
        print("Total words: {}".format(len(all_word_tuples)))
    elif args.mode == 'extract_lexicon':
        for x in all_word_tuples:
            #print(x[0], x[1], x[2])
            print(x[2])
    elif args.mode == 'get_unigrams':
        tt = get_list_of_unigrams('wordlist/nopunctuation_nocapitals_nolatins/sophia_lexicon.txt')
        print('A total of {} unique unigrams.'.format(len(tt)))
    else:
        raise NotImplementedError
