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

    return np.asarray(w), np.linalg.norm(w, ord=1)


def fit_model_gradient(x, y, epochs = 10000, beta = 0.05, eta = 1e-5,\
                       epsilon = 1e-5):
    err = []
    w = []
    w_norm = []

    x = np.asmatrix(x)
    # Start with all 0s
    w.append(np.zeros((x.shape[1])))
    w[-1] = w[-1].reshape([-1,1])
    for ii in range(epochs):
        alpha = eta/(ii*beta + 1)

        err.append(x*w[-1] - y)
        grad = x.T*err[-1]
        w.append(w[-1] - 2*alpha * grad)
        w_norm.append(np.linalg.norm((w[-1]), ord=1))
        if np.linalg.norm((w[-1] - w[-2]), ord=2) < epsilon:
            print('Epsilon value reached after {} epochs'.format(ii))
            return np.asarray(w[-1]),  w_norm
        if ii%100 == 0:
            print(ii)
            
    print('Epoch limit of {} reached'.format(epochs))

    return np.asarray(w[-1]), w_norm


def main():


    x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
    pp.preprocess(trn_len=10000, val_len=1000, tst_len=1000, mst_cmn_wd_len=160,\
               mst_cmn_wd_start = 0, order = 1)
    
#    x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
#    pp.normalize_all(x_trn, y_trn, x_val, y_val, x_tst, y_tst)

    w_gd, w_gd_norm = fit_model_gradient(x_trn, y_trn, epochs=10000, eta=1e-5)
    w_cf, w_cf_norm = fit_model_closed_form(x_trn, y_trn)

    w0 = np.zeros(len(w_cf))

    trn_w0 = get_error(x_trn, y_trn, w0)
    val_w0 = get_error(x_val, y_val, w0)

    trn_err_gd = get_error(x_trn, y_trn, w_gd)
    val_err_gd = get_error(x_val, y_val, w_gd)

    trn_err_cf = get_error(x_trn, y_trn, w_cf)
    val_err_cf = get_error(x_val, y_val, w_cf)

    figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(w_gd)
    plt.plot(w_cf)
    plt.title('Weights of Closed Form solution and Gradient Descent solution')
    plt.legend(['Gradient Descent','Closed Form'])
    plt.savefig('Figures/gd_{}_cf_{}.pdf'.format(val_err_gd, val_err_cf))
    plt.show()

#    figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
#    plt.plot(w_gd_norm)
##    plt.plot(np.ones(len(w_gd_norm))*w_cf_norm)
#    plt.title('Norm of the weights per epoch and closed form norm')
#    plt.show()

    print('GD Val Err: {}, GD Trn Err: {}\nCF Val Err: {}, CF Trn Err: {}'\
          '\nw0 Val Err: {}, w0 Trn Err: {}'.format(val_err_gd, \
                         trn_err_gd, val_err_cf, trn_err_cf, val_w0, trn_w0))
    return trn_err_cf, val_err_cf, trn_err_gd, val_err_gd

def cf_test():
    x_trn, y_trn, x_val, y_val, _, _ = \
    pp.preprocess(trn_len=10000, val_len=1000, tst_len=1000, mst_cmn_wd_len=160,\
               mst_cmn_wd_start = 0, order = 1)

    w_cf = fit_model_closed_form(x_trn, y_trn)
    trn_err_cf = get_error(x_trn, y_trn, w_cf)
    val_err_cf = get_error(x_val, y_val, w_cf)

    print('Trn Err: {}, Val Err: {}'.format(trn_err_cf, val_err_cf))

if __name__ == '__main__':
    trn_err_cf, val_err_cf, trn_err_gd, val_err_gd = main()

# 164 features
#GD Val Err: [0.99492486], GD Trn Err: [1.04980198]
#CF Val Err: [0.9950694], CF Trn Err: [1.04777632]
#w0 Val Err: [2.07317095], w0 Trn Err: [2.1124615]






