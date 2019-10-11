import re
import json


def doc_split(doc):
    # cut a long string into sentences
    # input: doc is a string longer than wd_size
    #      wd_size is a int
    # output: lst_doc = [piece_1, piece_2, ...], each piece is shorter than wd_size
    # 先考虑段落，太长的段落则考虑句子，以及striede
    lst_doc = list()
    # 使用re.finditer
    pattern = "[。；：;:]\n*"
    head = 0
    for match in re.finditer(pattern, doc):
        sentence_now = doc[head:match.end()]
        head = match.end()
        if len(sentence_now) > 0:
            lst_doc.append(sentence_now)
    return lst_doc
        

def window_split(lst_doc, max_length=300, window_ratio=0.5):
    # re-arange lst_doc into lst_text with overlapped sentences according to wd_stride
    # input: lst_doc = [sentence_1, sentence_2, ...]
    #      max_length is the maximum length of each piece
    #      window_ratio is the porpotion of the overlapped sentences between two pieces
    # output: lst_text = [piece_1, piece_2, ...], where piece_i = join(sentence_1, sentence_2, ...), len(piece_i) < wd_size
    if (not isinstance(window_ratio, float)) or window_ratio<0 or window_ratio>1:
        window_ratio = 0.5
    lst_text = list()
    n = len(lst_doc)  # number of sentences
    i = 0  # index of the current sentence in lst_doc
    while i < n:
        s = lst_doc[i]
        j = i + 1
        while (len(s) < max_length) and (j < n):
            s += lst_doc[j]
            j += 1  # update j
        lst_text.append(s)
        i += max(1, int(round(window_ratio*(j-i))))  # update i
    return lst_text
        

class EsRE_Pre(object):
    
    def __init__(self):
        pass
        
    def run(self, doc):
        lst_doc = doc_split(doc)
        lst_text = window_split(lst_doc, max_length=400, window_ratio=0.2)
        return lst_text
