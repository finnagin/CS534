# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:14:52 2019

@author: Michael
"""


# part 2.

import numpy as np
import matplotlib.pyplot as plt
import Helper_Class

myhelperclass = Helper_Class.Helper_Class()

# define weight vector for 5 elements

w = np.array([1,2,3,4,5])

# generate 30 training samples by randomly picking feature vector values and generating targets

X_train = 10*np.random.rand(30,5) 
y_train = np.dot(X_train, w) + np.random.normal(0,10, 30)

# generate 30 validation samples by randomly picking feature vector values and generating targets

X_val = 10*np.random.rand(30,5)
y_val = np.dot(X_val, w) + np.random.normal(0,10, 30)

w_0 = np.zeros(5)

myhelperclass = Helper_Class.Helper_Class()

# set stopping criterias

epsilon = .5

max_iter = 100000

# set learning rate - must be selected

alpha = .0001

# set initial weights to zero

X_train_shape = X_train.shape

w_0 = np.zeros(X_train_shape[1])

# define list of learning rates to try

lambda_list = [0, 10**(-3), 10**(-2), 10**(-1), 1, 10, 100]

# run descent algorithm using each learning rate and compute SSE for training and validation set

SSE_val_final = []
final_w_for_lambda_0 = []

for lambda_0 in lambda_list:
    
    # run gradient descent using given input values

    [w_vecs, w_grad_vecs, w_grad_norms] = myhelperclass.run_gradient_descent(y_train, X_train, w_0,alpha, lambda_0, epsilon, max_iter)
    
    # compute Sum of Square errors for training and test set for ever weight vector generated using 
    # gradient descent 
    
    SSE_train = myhelperclass.calculuate_SSE(w_vecs, y_train, X_train)
    SSE_val = myhelperclass.calculuate_SSE(w_vecs,y_val, X_val)

    # create list of ending weight vectors, 
    
    final_w_for_lambda_0.append(w_vecs[len(SSE_train)-1])
    
    # create integer based spacing to graph SSE for each weight vector associated
    # with a given iteration
    
    x_axis = np.linspace(0,len(SSE_train)-1,len(SSE_train))

    plt.figure()

    plt.plot(x_axis,SSE_train, x_axis, SSE_val)
    plt.title("SSE for training and validation data, alpha: " + str(alpha) + " lambda: " + str(lambda_0))
    plt.xlabel("training iteration")
    plt.ylabel("SSE")
    plt.legend(["SSE_Training", "SSE_Testing"])
    plt.show()

    plt.figure()

    plt.plot(x_axis, w_grad_norms)
    plt.title("Norm of gradient")
    plt.xlabel("training iteration")
    plt.ylabel("gradient norm")
    plt.show()
    
    # create list of final sum of squared errors
    
    SSE_val_final.append(SSE_val[len(SSE_val)-1])
    
# select best learning rate based upon best SSE for final weight using validation data
    
lambda_best = lambda_list[SSE_val_final.index(min(SSE_val_final))]
    
print("The best regularization constant based upon validation SSE is " + str(lambda_best))
    
# select best final weight vector using best SSE for final weights using validation data
    
w_best = final_w_for_lambda_0[SSE_val_final.index(min(SSE_val_final))]
    
print("The best final weight based upon validation SSE is" +str(w_best))
