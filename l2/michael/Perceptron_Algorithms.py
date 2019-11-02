# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:26:41 2019

@author: Michael
"""

import numpy as np

def OnlinePerceptron(Y_train, X_train, iters):

    (num_of_examples, feature_number) = X_train.shape
    
    W = np.zeros(feature_number)

    iter = 0;

    weight_list = []
    weight_list.append(W)

    while iter < iters:
        for t in range(num_of_examples):
            if Y_train[t]*np.dot(W,X_train[t,:]) <= 0:
                W = W + Y_train[t]*X_train[t,:]
        iter += 1 
        weight_list.append(np.copy(W))
    return weight_list

       

def AveragePerceptron(Y_train, X_train, iters):
   
    (num_of_examples, feature_number) = X_train.shape
    
    W = np.zeros(feature_number)
    W_ = np.zeros(feature_number)
    s = 1
    
    iter = 0;
    
    weight_list = []
    weight_list.append(W_)

    while iter < iters:
      for t in range(num_of_examples):
          if Y_train[t]*np.dot(W,X_train[t,:]) <= 0:
              W = W + Y_train[t]*X_train[t,:]
    
          W_ = (s*W_ + W)/(s+1)
          s = s + 1
      weight_list.append(np.copy(W_))
      iter += 1 
      

    return weight_list

def KernelizedPerceptron(Y_train, X_train, p, iters):
    
    (num_of_examples, feature_number) = X_train.shape
    
    a = np.zeros(num_of_examples)
    weight_list = []
    weight_list.append(np.copy(a))
    
    K = np.power(1 + np.matmul(X_train, np.transpose(X_train)),p)
    
    iter = 0
    
    while iter < iters:
        for t in range(num_of_examples):
            u = np.dot(a, np.transpose(K[:,t])*Y_train)
            if Y_train[t]*u <= 0:
                a[t] += 1
        weight_list.append(np.copy(a))
        iter += 1
                
    return weight_list           

def KernelizedPerceptronPrediction(weight, X_input, Y_train, X_train, p):
    
    K = np.power(1 + np.matmul(X_train, np.transpose(X_input)),p)
    
    predictions = np.sign(np.dot(np.transpose(K),weight*Y_train))
    
    return predictions
    
            

    