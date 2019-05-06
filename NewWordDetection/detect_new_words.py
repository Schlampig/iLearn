# -*- coding:utf-8 -*-
"""
Chinese word segmentation algorithm with corpus

Author:
    Xylander23 (https://github.com/xylander23/New-Word-Detection)
    Lyrichu (https://github.com/Lyrichu/NewWordDetection)

Reference:
    https://github.com/Moonshile/ChineseWordSegmentation
    http://www.matrix67.com/blog/archives/5044
    https://zlc1994.com/2017/01/04/

Modified by KuyiKing, 2019-05-06
"""

import os
import re
import math
import time
import codecs
import pandas as pd
from collections import Counter
import jieba


# Calculate entropy
####################################################################################################
def compute_entropy(_list):
    # left/right neighbors entropy for new word detection
    length = float(len(_list))
    freq = {}
    if length == 0:
        return 0
    else:
        for i in _list:
            freq[i] = freq.get(i, 0) + 1
        return sum(map(lambda x: - x / length * math.log(x / length), freq.values()))


def extract_cadicateword(_doc, _max_word_len):
    indexes = []
    doc_length = len(_doc)
    for i in range(doc_length):
        for j in range(i + 1, min(i + 1 + _max_word_len, doc_length + 1)):
            indexes.append((i, j))
    return sorted(indexes, key=lambda _word: _doc[_word[0]:_word[1]])  # 字典序排列


def gen_bigram(_word_str):
    # A word is divide into two part by following all possible combines.
    # For instance, ABB can divide into (a,bb),(ab,b)
    # 产生一个word所有可能的二元划分集合
    return [(_word_str[0:_i], _word_str[_i:]) for _i in range(1, len(_word_str))]


# Split documentation
####################################################################################################
class GetWord(object):
    # Record every candidate word information include left neighbors, right neighbors, frequency, PMI

    def __init__(self, text):
        super(GetWord, self).__init__()
        self.text = text  # 候选词
        self.freq = 0.0  # 候选词出现的频率
        self.left = []  # record left neighbors
        self.right = []  # record right neighbors
        self.pmi = 0  # 凝聚度

    def update_data(self, left, right):
        self.freq += 1.0  # 候选词出现的次数加1
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def compute_indexes(self, length):
        # compute frequency of word,and left/right entropy
        # length是整个doc的长度
        self.freq /= length
        self.left = compute_entropy(self.left)
        self.right = compute_entropy(self.right)

    def compute_pmi(self, words_dict):
        # 这里的words_dict是word_cad
        # key:word,value:word_info
        # compute all kinds of combines for word
        sub_part = gen_bigram(self.text)
        if len(sub_part) > 0:
            # 使用一个具体的例子来概括就是:
            # 计算min{p(电影院)/(p(电影)*p(院)),p(电影院)/(p(电)*p(影院))}
            self.pmi = min(
                map(lambda word: math.log(self.freq / words_dict[word[0]].freq / words_dict[word[1]].freq), sub_part))


class SegDoc(object):
    # Main class for Chinese word segmentation
    # 1. Generate words from a long enough document
    # 2. Do the segmentation work with the document
    def __init__(self, doc, max_word_len=5, min_tf=0.000005, min_entropy=0.07, min_pmi=6.0):
        super(SegDoc, self).__init__()
        self.max_word_len = max_word_len  # 最大的词长度
        self.min_tf = min_tf  # 最小的word term frequency
        self.min_entropy = min_entropy
        self.min_pmi = min_pmi  # 最小的凝聚度
        # analysis documents
        self.word_info = self.gen_words(doc)
        count = float(len(self.word_info))  # 所有word的个数
        self.avg_frq = sum(map(lambda w: w.freq, self.word_info)) / count
        self.avg_entropy = sum(map(lambda w: min(w.left, w.right), self.word_info)) / count
        self.avg_pmi = sum(map(lambda w: w.pmi, self.word_info)) / count
        # 匿名过滤函数
        filter_function = lambda f: len(f.text) > 1 and f.pmi > self.min_pmi and f.freq > self.min_tf \
                                    and min(f.left, f.right) > self.min_entropy
        self.word_tf_pmi_ent = map(lambda w: (w.text, len(w.text), w.freq, w.pmi, min(w.left, w.right)),
                                   filter(filter_function, self.word_info))

    def gen_words(self, doc):
        word_index = extract_cadicateword(doc, self.max_word_len)
        word_cad = {}  # 候选词字典
        for suffix in word_index:
            word = doc[suffix[0]:suffix[1]]  # 候选词
            if word not in word_cad:
                word_cad[word] = GetWord(word)
                # record frequency of word and left neighbors and right neighbors
            word_cad[word].update_data(doc[suffix[0] - 1:suffix[0]], doc[suffix[1]:suffix[1] + 1])
        length = len(doc)
        # computing frequency of candicate word and entropy of left/right neighbors
        for word in word_cad:
            word_cad[word].compute_indexes(length)
        # ranking by length of word
        values = sorted(word_cad.values(), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_pmi(word_cad)
        # ranking by frequency
        return sorted(values, key=lambda v: len(v.text), reverse=False)


# Post-process
####################################################################################################
def clean_words(doc):
    # input: doc is a long string
    # output: doc is a cleaned long string
    pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')  # 要去除的无意义的符号
    doc = pattern.sub(r'', doc)  # 替换为空格
    return doc


def filter_words(word, stop_path, doc, filter_exist=True):
    # input: word, object
    #        stop_path, string
    #        doc, clean long string
    #        filter_exist, if True delete existed words
    # output: word_list = [word_item, word_item, ...], where word_item = (word, length, freq, pmi, entropy)
    # generate dictionary for existing words
    dict_doc = dict()
    if filter_exist:
        dict_doc = dict(Counter(jieba.lcut(doc)))
    # load list_stop
    list_stop = list()
    if os.path.exists(stop_path):
        for i in codecs.open(stop_path, 'r', "utf-8"):
            list_stop.append(i.split(' ')[0].replace("\n", ""))
    # filter stop word
    word_list = list()
    for i in word.word_tf_pmi_ent:
        if (i[0] not in list_stop) and (i[0] not in dict_doc):
            word_list.append([i[0], i[1], i[2], i[3], i[4]])
    return word_list


def rank_words(word_list):
    # ranking on entropy (primary key) and pmi (secondary key)
    # input & output: word_list = [word_item, word_item, ...], where word_item = (word, length, freq, pmi, entropy)
    word_list = sorted(word_list, key=lambda word: word[3], reverse=True)
    word_list = sorted(word_list, key=lambda word: word[4], reverse=True)
    return word_list


def save_words(word_list, save_path):
    # input: word_list = [word_item, word_item, ...], where word_item = (word, length, freq, pmi, entropy)
    #        save_path is a string
    # output: None
    print("New words saving ...")
    seg = pd.DataFrame(word_list, columns=['word', 'length', 'fre', 'pmi', 'entropy'])
    seg.to_csv(save_path, index=False, encoding="utf-8")
    print("New words saved.")
    return None


def discover_words(load_path, save_path, stop_path, filter_exist=False):
    # input: load_path, string for document path
    #        stop_path, string
    #        filter_exist, if True delete existed words
    # output: boolean
    # document splitting
    assert load_path.endswith(".txt")
    assert save_path.endswith(".csv")
    assert stop_path.endswith(".txt")
    doc = codecs.open(load_path, "r", "utf-8").read()
    doc = clean_words(doc)
    word = SegDoc(doc, max_word_len=3, min_tf=1e-08, min_entropy=1.0, min_pmi=3.0)
    print("Results: avg_frq-{}, avg_pmi-{}, avg_entropy-{}.".format(word.avg_frq, word.avg_pmi, word.avg_entropy))
    # word filtering
    word_list = filter_words(word, stop_path, doc, filter_exist)
    if len(word_list) == 0:
        return False
    word_list = rank_words(word_list)
    save_words(word_list, save_path)
    return True


# Run
####################################################################################################
if __name__ == "__main__":
    print("Start to analyze ...")
    start_time = time.clock()
    load_path = "document.txt"
    save_path = "results.csv"
    stop_path = "dict.txt"
    discover_words(load_path, save_path, stop_path, True)
    duration = time.clock() - start_time
    print("Time-cost: {}.".format(duration))
    print("Finished.")
