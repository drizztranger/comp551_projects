# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:04:48 2019
Mini Project 1, data creation
@author: sgeoff1
"""

import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
from collections import Counter

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment  

def data_preprocessing(data, unwanted_chars = ('\r', '"', "'", '(', ')', \
                                               '*', ':', ';','[', ']', '_',\
                                               '\xbb', '\xbf', '\xef','®',\
                                               '=', '\n', '0', '1','2','3',\
                                               '4','5','6','7','8','9','^',\
                                               '—','\\','/', '?', '!', '.')):
    word_list = []
    for item in data:
        # Remove all special characters
        item['text'] = "".join(c for c in item['text'] if c not in unwanted_chars)
        item['text'] = item['text'].lower().split()
        word_list = word_list + item['text']   
    return data, word_list

def feature_creation(data, mst_cmn_wds):
    for item in data:
        item['avg_word_len'] = sum(len(word) for word in item['text'] ) / (len(item['text']) + 1)
        item['word_count'] = len(item['text'])
        item['pop_contr_interact'] = item['controversiality']*item['popularity_score']
        item['child_pop_interact'] = item['children']*item['popularity_score']
        item['wc_avg_len_interact'] = item['avg_word_len']*item['word_count']
        if item['is_root'] == True:
            item['is_root'] = 1    
        else:
            item['is_root'] = 0
        item['mst_cmn_wds'] = list(np.zeros(len(mst_cmn_wds)))
        for ii, wd in enumerate(mst_cmn_wds):
            item['mst_cmn_wds'][ii] = len([x for x in item['text'] if wd[0] == x])
    return data

def word_appearances(word_list):
    term_appearance = Counter(word_list)   
    return term_appearance

def separate_data(data):
    x = []
    y = []
    key = data[0].keys()
    for item in data:
        features = []
        for k in key:
            if k != 'text' and k != 'popularity_score':
                features.append(item[k])
            if k == 'popularity_score':
                target = item[k]
        x.append(features)
        y.append(target)
    return x, y

data, word_list = data_preprocessing(data)
term_appearance = word_appearances(word_list)
mst_cmn_wds = term_appearance.most_common()[0:10]
data = feature_creation(data, mst_cmn_wds)
x, y = separate_data(data)

print(data[1])
