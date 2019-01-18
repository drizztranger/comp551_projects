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
        for wd in mst_cmn_wds:
            item[wd[0]]= len([x for x in item['text'] if wd[0] == x])
            
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
        # Also append an intersect term to x
        features.append(1)
        x.append(features)
        y.append(target)
    
    # make features into np.array
    x = np.array(x)
    # Get array of targets
    y = np.array(y)
    y = y.reshape([-1,1])
    return x, y

def preprocess(trn_len=200, val_len=50, tst_len=50, mst_cmn_wd_len = 10):
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
    mst_cmn_wds = term_appearance.most_common()[0:mst_cmn_wd_len]
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
    main()
