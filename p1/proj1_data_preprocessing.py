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
def data_preprocessing(data, unwanted_chars = ('')):
    word_list = []
    for item in data:
        # Remove all special characters
        item['text'] = "".join(c for c in item['text'] if c not in unwanted_chars)
        item['text_lower_split'] = item['text'].lower().split()
        word_list = word_list + item['text_lower_split']
    return data, word_list

# Create features
    # This is pretty hardcoded for now
def feature_creation(data, mst_cmn_wds):
#    for item in data:
#        # Get the length of the set of characters in this comment
#        item['diff_char_len'] = len(set(item['text']))
#        # Get the number of characters
##        item['char_num'] = len(item['text'])
#        # Get avg word length
#        item['avg_word_len'] = sum(len(word) for word in item['text_lower_split'] ) / (len(item['text_lower_split']) + 1)
##       # Get word count
#        item['word_count'] = len(item['text_lower_split'])
##       # Multiply word count with average word length
#        item['wc_avg_len_interact'] = item['avg_word_len']*item['word_count']
##        # Multiply is_root with childrens
#        item['ir_child_interact'] = item['is_root']*item['children']
##        # Interact interact
#        item['inter_inter'] = item['wc_avg_len_interact'] * item['ir_child_interact']

#        # Create most common word vector and fill it with current text
#        for wd in mst_cmn_wds:
#            item[wd[0]]= len([x for x in item['text_lower_split'] if wd[0] == x])
#
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
            if k != 'text' and k != 'popularity_score' and k != 'text_lower_split':
                features.append(item[k])
            # Make target the popularity score
            if k == 'popularity_score':
                target = item[k]
        # Also append an intersect term to x
#        features.append(1)
        x.append(features)
        y.append(target)

    # make features into np.array
    x = np.array(x)
    # Get array of targets
    y = np.array(y)
    y = y.reshape([-1,1])
    return x, y

# Find the min and max values for each column
def minmax(data):
    minmax = list()
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        value_min = min(col)
        value_max = max(col)
        if value_min == value_max:
            value_max += 1
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range [0, 1]
def normalize(data, minmax):
    for row in data:
        for i in range(len(row)):
            if minmax[i][1] != minmax[i][0]:
                row[i] = ((row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))
            else:
                row[i] = ((row[i] - minmax[i][0]) / (1 + minmax[i][1] - minmax[i][0]))


def preprocess(trn_len=200, val_len=50, tst_len=50, mst_cmn_wd_len = 10,\
               mst_cmn_wd_start = 10):
    # Load the data
    with open("proj1_data.json") as fp:
        data = json.load(fp)

    data_trn = data[0:trn_len]
    data_val = data[trn_len:trn_len + val_len]
    data_tst = data[trn_len + val_len:trn_len + val_len + tst_len]

    # Preprocess data, create word list (also makes the unwanted chars go away)
    data_trn, word_list = data_preprocessing(data_trn, unwanted_chars = ('.'))
    data_val, _ = data_preprocessing(data_val, unwanted_chars = ('.'))
    data_tst, _ = data_preprocessing(data_tst, unwanted_chars = ('.'))

    # Count appearances of each word present in training set
    term_appearance = word_appearances(word_list)
    # get the 160 most common words present in training set
    mst_cmn_wds = term_appearance.most_common()[mst_cmn_wd_start:\
                                             mst_cmn_wd_start+mst_cmn_wd_len]
    # Create features
    data_trn = feature_creation(data_trn, mst_cmn_wds)
    data_val = feature_creation(data_val, mst_cmn_wds)
    data_tst = feature_creation(data_tst, mst_cmn_wds)

    # Separate the data into features and targets
    x_trn, y_trn = separate_data(data_trn)
    x_val, y_val = separate_data(data_val)
    x_tst, y_tst = separate_data(data_tst)

    # return np.array of training set, validation set, and test set
    return x_trn, y_trn, x_val, y_val, x_tst, y_tst

def preprocess_normalize_data(trn_len = 1000, val_len = 1000, tst_len = 1000,\
                              mst_cmn_wrds = 60, mst_cmn_start = 5):
    # Get 1d array of inputs and targets
    x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
    preprocess(trn_len = trn_len, val_len = val_len, tst_len = tst_len, \
                   mst_cmn_wd_len = mst_cmn_wrds, mst_cmn_wd_start = mst_cmn_start)

    normalize(x_trn, minmax(x_trn))
    normalize(y_trn, minmax(y_trn))

    normalize(x_val, minmax(x_val))
    normalize(y_val, minmax(y_val))

    normalize(x_tst, minmax(x_tst))
    normalize(y_tst, minmax(y_tst))


    return x_trn, y_trn, x_val, y_val, x_tst, y_tst

def main():
        # Load the data
    with open("proj1_data.json") as fp:
        data = json.load(fp)

    data = data[0:100]
    # Preprocess data, create word list (also makes the unwanted chars empty)
    data, word_list = data_preprocessing(data, unwanted_chars = ('.'))
    # Count appearances of each word
    term_appearance = word_appearances(word_list)
    # get the 160 most common words
    mst_cmn_wds = term_appearance.most_common()[0:160]
    # Create features
    data = feature_creation(data, mst_cmn_wds)
    # Separate the data into features and targets
    x, y = separate_data(data)
    return x, y, data, mst_cmn_wds

if __name__ == '__main__':
    x, y, data, mst_cmn_wds = main()
    normalize(x, minmax(x))
    normalize(y, minmax(y))














