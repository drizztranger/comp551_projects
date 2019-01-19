# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:36:10 2019
Closed form linear regression of reddit comments
@author: sgeoff1
"""
from numpy import linalg
import proj1_data_preprocessing as pp

def get_error(x, y, w):
    err = 0
    for ii, instance in enumerate(x):
        yhat = w.T.dot(instance)
        err += (y[ii] - yhat)**2
    return err/len(x)

unwanted_chars = ('\r', '"', "'", '(', ')','*', ':', ';','[', ']', '_',\
                  '\xbb', '\xbf', '\xef','®','=', '\n', '0', '1','2','3',\
                  '4','5','6','7','8','9','^','—','\\','/', '?', '!', '.')

# Get 1d array of inputs and targets
x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
 pp.preprocess(trn_len = 10000, val_len = 1000,tst_len = 1000, \
               mst_cmn_wd_len = 60)

# Closed form lin reg
xtx = linalg.inv(x_trn.transpose().dot(x_trn))
xty = x_trn.transpose().dot(y_trn)
w = xtx.dot(xty)

trn_err = get_error(x_trn, y_trn, w)
val_err = get_error(x_val, y_val, w)

print('# Training Error: {} Validation Error: {}'.format(trn_err, val_err))


# No extra features:
#Training Error
#[1.08204066]
#Validation Error
#[1.01495091]

# with errythin
#Training Error
#[1.05891058]
#Validation Error
#[0.97757024]

# Added interact-interact term
# Training Error: [1.05661095] Validation Error: [0.98815119] OVERFITTING?

# W\O char num and dif char num
# Training Error: [1.05818106] Validation Error: [0.99185777]

      
# mst_cmn_wds (160):
#Training Error
#[1.04815463]
#Validation Error
#[0.99408643]

# mst_cmn_wds (60):
#Training Error
#[1.06037456]
#Validation Error
#[0.98650913]

# Just most common words (160)
#Training Error
#[1.04832305]
#Validation Error
#[0.99403616]

