# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:36:10 2019
Closed form linear regression of reddit comments
@author: sgeoff1
"""
from numpy import linalg
import proj1_data_preprocessing as pp
import matplotlib.pyplot as plt

def get_error(x, y, w):
    err = 0
    for ii, instance in enumerate(x):
        yhat = w.T.dot(instance)
        err += (y[ii] - yhat)**2
    return err/len(x)

def main():

    tot_trn_err = []
    tot_val_err = []
    for mst_cmn_wrds in range(10, 170, 20):
        validation_err = []
        training_err = []
        for mst_cmn_start in range(0, 80, 10):
            # Get 1d array of inputs and targets
            x_trn, y_trn, x_val, y_val, x_tst, y_tst = \
             pp.preprocess(trn_len = 10000, val_len = 1000, tst_len = 1000, \
                           mst_cmn_wd_len = mst_cmn_wrds, mst_cmn_wd_start = mst_cmn_start)
             
            pp.normalize(x_trn, pp.minmax(x_trn))
            pp.normalize(y_trn, pp.minmax(y_trn))
            
            pp.normalize(x_val, pp.minmax(x_val))
            pp.normalize(y_val, pp.minmax(y_val))
            
            # Closed form lin reg
            xtx = linalg.inv(x_trn.transpose().dot(x_trn))
            xty = x_trn.transpose().dot(y_trn)
            w = xtx.dot(xty)
            
            trn_err = get_error(x_trn, y_trn, w)
            val_err = get_error(x_val, y_val, w)
            
            print('# Training Error: {} Validation Error: {}. Most Common word start: {}, Most common word len: {}'\
                  .format(trn_err, val_err, mst_cmn_start, mst_cmn_wrds))
            validation_err.append(val_err)
            training_err.append(trn_err)
            
        tot_trn_err.append(training_err)
        tot_val_err.append(validation_err)
            
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    