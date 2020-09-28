# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:48:44 2020

@author: Haojun666
"""
import numpy as np

def compute_mse(y, tx, w):
    '''
    calculate the mean square error
    '''
    e = y - tx.dot(w)
    return 0.5*np.mean(e * e)

def compute_gradient(y, tx, w):
    '''
    compute the gradient
    '''
    N = tx.shape[0]
    e = y - np.dot(tx, w)
    return -(1/N)*np.dot(tx.T, e)

def compute_accuracy(y, y_pred):
    '''
    calculate the accuracy of the prediction
    '''
    count = 0
    for ind, value in enumerate(y_pred):
        if value == y[ind]:
            count += 1
    return count/len(y)

def build_poly(tX, degree):
    temp_dict = {}
    count = 0
    for i in range(tX.shape[1]):
        for j in range(i+1,tX.shape[1]):
            temp = tX[:,i] * tX[:,j]
            temp_dict[count] = [temp]
            count += 1
    poly_length = tX.shape[1] * (degree + 1) + count + 1
    poly = np.zeros(shape = (tX.shape[0], poly_length))
    for deg in range(1,degree+1):
        for i in range(tX.shape[1]):
            poly[:,i + (deg-1) * tX.shape[1]] = np.power(tX[:,i],deg)
    for i in range(count):
        poly[:, tX.shape[1] * degree + i] = temp_dict[i][0]
    for i in range(tX.shape[1]):
        poly[:,i + tX.shape[1] * degree + count] = np.abs(tX[:,i])**0.5
    return poly

def least_squares_GD(y, tX, initial_w, max_iters, gamma):
    '''
    implement the linear regression with gradient descent
    '''
    w = initial_w
    
    for n_iter in range(max_iters):
        l = compute_mse(y, tX, w)
        g = compute_gradient(y, tX, w)
        w -= gamma * g
        print(l)
    return l, w

def least_squares_SGD(y, tX, initial_w, max_iters,gamma):
    '''
    implement the linear regression with stochastic gradient descent
    '''
    w = initial_w
    return 0

def least_squares(y, tX):
    '''
    implement the least squares regression
    '''
    a = np.dot(tX.T, tX)
    b = np.dot(tX.T, y)
    w = np.linalg.solve(a, b)   
    return w

def ridge_regression(y, tX, lambda_):
    '''
    ridge regression
    '''
    N = len(y)
    D = tX.shape[1]
    lambda_ = 2 * N * lambda_
    a = np.dot(tX.T, tX) + lambda_ * np.identity(D)
    b = np.dot(tX.T, y)
    w = np.linalg.solve(a, b)
    return w

