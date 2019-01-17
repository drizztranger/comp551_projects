# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:04:48 2019
Mini Project 1, data creation
@author: sgeoff1
"""

import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
from collections import Counter

# Preprocess the data, rm all unwanted chars, lowercase, and split words. Also 
# Also appends every comments in one big list for most common word vector creation
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

# Create features
    # This is pretty hardcoded for now
def feature_creation(data, mst_cmn_wds):
    for item in data:
        # make is_root 1 or 0
        if item['is_root'] == True:
            item['is_root'] = 1    
        else:
            item['is_root'] = 0
            
        # Get avg word length
        item['avg_word_len'] = sum(len(word) for word in item['text'] ) / (len(item['text']) + 1)
        # Get word count
        item['word_count'] = len(item['text'])
        # Multiply word count with average word length
        item['wc_avg_len_interact'] = item['avg_word_len']*item['word_count']
        # Multiply is_root with childrens
        item['ir_child_interact'] = item['is_root']*item['children']
        
        # Create most common word vector and fill it with current text
        item['mst_cmn_wds'] = list(np.zeros(len(mst_cmn_wds)))
        for ii, wd in enumerate(mst_cmn_wds):
            item['mst_cmn_wds'][ii] = len([x for x in item['text'] if wd[0] == x])
            
    return data

# Count the number of times each word appears 
def word_appearances(word_list):
    term_appearance = Counter(word_list)   
    return term_appearance

# Separate the data into the features and the target
    # This is pretty hard coded for now
def separate_data(data):
    x = []
    y = []
    # Get the keys of the dictionary
    key = data[0].keys()
    for item in data:
        features = []
        for k in key:
            # We dont want the text or the target for our features
            if k != 'text' and k != 'popularity_score':
                features.append(item[k])
            # Make target the popularity score
            if k == 'popularity_score':
                target = item[k]
        x.append(features)
        y.append(target)
    return x, y


# Load the data
with open("proj1_data.json") as fp:
    data = json.load(fp)

# Preprocess data, create word list (also makes the unwanted chars empty)
data, word_list = data_preprocessing(data, unwanted_chars = ('.'))
# Count appearances of each word
term_appearance = word_appearances(word_list)
# get the 160 most common words
mst_cmn_wds = term_appearance.most_common()[0:10]
# Create features
data = feature_creation(data, mst_cmn_wds)
# Separate the data into features and targets
x, y = separate_data(data)
# Print the first instance and target
print(x[0])
print(y[0])
