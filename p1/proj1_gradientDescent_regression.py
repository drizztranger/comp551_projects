#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:18:19 2019

@author: simon

Gradient Descent Version of Linear Regression
"""

import proj1_data_preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import linalg
from matplotlib.pyplot import figure

def get_error(x, y, w):
    err = 0
    for ii, instance in enumerate(x):
        yhat = w.T.dot(instance)
        err += (y[ii] - yhat)**2
    return err/len(x)

def fit_model_closed_form(x, y):
    x = np.asmatrix(x)
    xtx = x.T*x

    # Closed form lin reg, checks if invertible
    if linalg.cond(xtx) < 1/sys.float_info.epsilon:
        xtxi = xtx.I
        xty = x.T*y
        w = xtxi*xty
    else:
        #handle it
        print('Singular matrix, closed form not solvable')
        w = np.zeros((x.shape[1], 1))

    return w


def fit_model_gradient(x, y, epochs = 1000000, beta = 0.1, eta = 1e-5,\
                       epsilon = 1e-7):
    err = []
    w = []

    x = np.asmatrix(x)
    # Start with all 0s
    w.append(np.zeros((x.shape[1])))
    w[-1] = w[-1].reshape([-1,1])
    for ii in range(epochs):
        alpha = eta/(ii*beta + 1)

        err.append(x*w[-1] - y)
        grad = x.T*err[-1]
        w.append(w[-1] - 2*alpha * grad)
        if np.linalg.norm((w[-1] - w[-2]), ord=2) < epsilon:
            print('Epsilon value reached after {} epochs'.format(ii))
            return np.asarray(w[-1])

    print('Epoch limit of {} reached'.format(epochs))

    return np.asarray(w[-1])

def preprocess_normalize_data(trn_len = 1000, val_len = 1000, tst_len = 1000,\
                              mst_cmn_wrds = 60, mst_cmn_start = 5):
    # Get 1d array of inputs and targets
    x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
     pp.preprocess(trn_len = 1000, val_len = 100, tst_len = 100, \
                   mst_cmn_wd_len = mst_cmn_wrds, mst_cmn_wd_start = mst_cmn_start)

    pp.normalize(x_trn, pp.minmax(x_trn))
    pp.normalize(y_trn, pp.minmax(y_trn))

    pp.normalize(x_val, pp.minmax(x_val))
    pp.normalize(y_val, pp.minmax(y_val))

    pp.normalize(x_tst, pp.minmax(x_tst))
    pp.normalize(y_tst, pp.minmax(y_tst))


    return x_trn, y_trn, x_val, y_val, x_tst, y_tst


def main():


    x_trn, y_trn, x_val, y_val, _, _ = \
    preprocess_normalize_data(trn_len = 10000, val_len = 1000, mst_cmn_wrds = 160)

    w_gd = fit_model_gradient(x_trn, y_trn)
    w_cf = fit_model_closed_form(x_trn, y_trn)

    trn_err_gd = get_error(x_trn, y_trn, w_gd)
    val_err_gd = get_error(x_val, y_val, w_gd)

    trn_err_cf = get_error(x_trn, y_trn, w_cf)
    val_err_cf = get_error(x_val, y_val, w_cf)

    figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(w_gd)
    plt.plot(w_cf)
    plt.title('Weights of both Closed Form sln and Gradient Descent sln')
    plt.legend(['Gradient Descent','Closed Form'])
    return trn_err_cf, val_err_cf, trn_err_gd, val_err_gd

if __name__ == '__main__':
    trn_err_cf, val_err_cf, trn_err_gd, val_err_gd = main()
    print('GD Val Err: {}, CF Val Err: {}'.format(val_err_gd, val_err_cf))









