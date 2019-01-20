# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:36:10 2019
Closed form linear regression of reddit comments
@author: sgeoff1
"""
from numpy import linalg
import proj1_data_preprocessing as pp
import matplotlib.pyplot as plt
import sys
import numpy as np

def get_error(x, y, w):
    err = 0
    for ii, instance in enumerate(x):
        yhat = w.T.dot(instance)
        err += (y[ii] - yhat)**2
    return err/len(x)

def main():
    lw_st = 0
    lw_end = 40
    lw_jump = 2

    tot_trn_err = []
    tot_val_err = []
    for mst_cmn_wrds in range(30, 170, 2):
        validation_err = []
        training_err = []
        del(validation_err)
        del(training_err)
        training_err=[None]*(len(range(lw_st, lw_end, lw_jump)))
        validation_err=[None]*(len(range(lw_st, lw_end, lw_jump)))

        for ii, mst_cmn_start in enumerate(range(lw_st, lw_end, lw_jump)):

            # Get 1d array of inputs and targets
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
             pp.preprocess(trn_len = 10000, val_len = 1000, tst_len = 1000, \
                           mst_cmn_wd_len = mst_cmn_wrds, mst_cmn_wd_start = mst_cmn_start)

            pp.normalize(x_trn, pp.minmax(x_trn))
            pp.normalize(y_trn, pp.minmax(y_trn))

            pp.normalize(x_val, pp.minmax(x_val))
            pp.normalize(y_val, pp.minmax(y_val))

            xtx = x_trn.transpose().dot(x_trn)
            # Closed form lin reg, checks if its invertible
            if linalg.cond(xtx) < 1/sys.float_info.epsilon:
                xtxi = linalg.inv(xtx)
            else:
                #handle it
                print('Singular matrix, going to next mst_cmn_wds value')
                tot_trn_err.append(training_err)
                tot_val_err.append(validation_err)
                break

            xty = x_trn.transpose().dot(y_trn)
            w = xtxi.dot(xty)

            trn_err = get_error(x_trn, y_trn, w)
            val_err = get_error(x_val, y_val, w)

#            print('# Training Error: {} Validation Error: {}. Most Common word start: {}, Most common word len: {}'\
#                  .format(trn_err[0], val_err[0], mst_cmn_start, mst_cmn_wrds))

            validation_err[ii] = val_err
            training_err[ii] = trn_err
#            print(validation_err)
            print('{} done out of {} for mst cmn words {}'.format(\
                  ii, len(range(lw_st, lw_end, lw_jump)), mst_cmn_wrds))
        tot_trn_err.append(training_err)
        tot_val_err.append(validation_err)

    for wd_cnt_err in range(len(tot_trn_err)):

        plt.plot(range(lw_st, lw_end, lw_jump), tot_val_err[wd_cnt_err])
        plt.plot(range(lw_st, lw_end, lw_jump), tot_trn_err[wd_cnt_err])
        plt.title('{} most common words'.format(range(30,170,2)[wd_cnt_err]))
        plt.xlabel('Most Common words ignored')
        plt.ylabel('Error (y - yhat)^2')
        fname='{}_most_common_words_error.pdf'.format(range(30,170,2)[wd_cnt_err])
        plt.savefig(fname)
        plt.show()

    return tot_val_err, tot_trn_err

if __name__ == '__main__':
    tot_val_err, tot_trn_err = main()



# No extra features, normalized x and y, 60 mst cmn wds
# Training Error: [0.15660152] Validation Error: [0.11444731]

# All the extra features, normalized x and y 60, mst cmn wds
# Training Error: [0.02674199] Validation Error: [0.10225587]

# No wd diff cnt, char count, normalized x and y, 60 mst cmn wds
# Training Error: [0.04945909] Validation Error: [0.36597512] (way worse?)

# no char count, normalized x and y, 60 mst cmn wds
# Training Error: [0.0267645] Validation Error: [0.10045445] (BEST)

# No wd diff cnty, normalized x and y, 60 mst cmn wds
# Training Error: [0.04907331] Validation Error: [0.32440512] (also worse)

# Weights = [0...0] is err = 0.1771

















