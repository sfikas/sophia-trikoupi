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
        strip_punctuation=True,
        force_lowercase=False):
    """
    Returns a list of tuples. Each tuple corresponds to one word.
    """
    if force_lowercase is True:
        raise NotImplementedError
    with open(xmlname, 'r') as f:
        xmldata = f.read().replace('\n', '')
    rexp = '<Word id="(.*?)">\s*<Coords points="(.*?)"/>\s*<TextEquiv>\s*<Unicode>(.+?)</Unicode>\s*</TextEquiv>\s*</Word>'
    words = re.findall(rexp, xmldata)
    #
    words_processed = []
    punctuation_mark_table = dict.fromkeys(map(ord, '\'‘&,.’:;"-()!·'), None)
    for x in words:
        if strip_punctuation is True:
            transcr_new = x[2].translate(punctuation_mark_table)
        else:
            transcr_new = x[2]
        transcr_new = transcr_new.replace('&quot', '')
        transcr_new = transcr_new.replace('quot', '')
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

def read_sophia_xmls():
    all_word_tuples = []
    all_xmls = []
    for x in range(1, 48):
        if x == 12:
            continue #Page 12 was omitted / doesn't exist
        all_xmls.append('data/_00{0:02d}.xml'.format(x))
    for x in all_xmls:
        all_word_tuples += get_words_from_pagexml(x)
    return all_word_tuples

if __name__=='__main__':
    logger = logging.getLogger('SophiaTrikoupi::dataset')
    logger.info('------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-mode', required=True, choices=['check','extract_lexicon','get_unigrams'], help='Execution mode.')
    args = parser.parse_args()

    all_word_tuples = read_sophia_xmls()

    if args.mode == 'check':
        print("Total words: {}".format(len(all_word_tuples)))
    elif args.mode == 'extract_lexicon':
        for x in all_word_tuples:
            #print(x[0], x[1], x[2])
            print(x[2])
    elif args.mode == 'get_unigrams':
        get_list_of_unigrams('sophia_lexicon_punctation_removed.txt')
    else:
        raise NotImplementedError
