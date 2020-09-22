# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:48:44 2020

@author: Haojun666
"""
import numpy as np

def compute_mse(y, tx, w):
    '''calculate the mean square error'''
    e = y - tx.dot(w)
    return 0.5*np.mean(e * e)

def compute_gradient(y, tx, w):
    '''compute the gradient'''
    N = tx.shape[0]
    e = y - np.dot(tx, w)
    return -(1/N)*np.dot(tx.T, e)
