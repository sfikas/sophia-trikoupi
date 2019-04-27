import re
import argparse
import os

def get_words_from_pagexml(xmlname):
    """
    Returns a list of tuples. Each tuple corresponds to one word.
    """
    with open(xmlname, 'r') as f:
        xmldata = f.read().replace('\n', '')
    rexp = '<Word id="(.*?)">\s*<Coords points="(.*?)"/>\s*<TextEquiv>\s*<Unicode>(.+?)</Unicode>\s*</TextEquiv>\s*</Word>'
    words = re.findall(rexp, xmldata)
    print('Found {} words in file {}.'.format(len(words), xmlname))
    return(words)

if __name__=='__main__':
    all_word_tuples = []
    all_xmls = []
    for x in range(1, 48):
        if x == 12:
            continue #Page 12 was omitted / doesn't exist
        all_xmls.append('_00{0:02d}.xml'.format(x))
    for x in all_xmls:
        all_word_tuples += get_words_from_pagexml(x)
    print("Total words: {}".format(len(all_word_tuples)))
