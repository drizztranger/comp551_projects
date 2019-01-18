# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:36:10 2019
Closed form linear regression of reddit comments
@author: sgeoff1
"""
# INITIAL CLOSED FORM SLN FOR SIMPLE DATA
from numpy import linalg
import proj1_data_preprocessing as pp

# Get 1d array of inputs and targets
x_trn, y_trn, x_val, y_val, x_tst, y_tst =  pp.preprocess(trn_len = 1000)

# Closed form lin reg
xtx = linalg.inv(x_trn.transpose().dot(x_trn))
xty = x_trn.transpose().dot(y_trn)
w = xtx.dot(xty)

err = 0
for ii, instance in enumerate(x_val):
    yhat = w.T.dot(instance)
    err += (y_val[ii] - yhat)**2
err = err/len(x_val)
print(err)
