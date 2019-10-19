# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:14:52 2019

@author: Michael
"""

#script demonstrating Helper_Class helper functions

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

alpha = .0001

lambda_0 = 1

epsilon = .01

max_iter = 10000

[w_vecs, w_grad_vecs, w_grad_norms] = myhelperclass.run_gradient_descent(y_train, X_train, w_0,alpha, lambda_0, epsilon, max_iter)

SSE_train = myhelperclass.calculuate_SSE(w_vecs, y_train, X_train)
SSE_val = myhelperclass.calculuate_SSE(w_vecs,y_val, X_val)

x_axis = np.linspace(0,len(SSE_train),len(SSE_train))

plt.figure(0)

plt.plot(x_axis,SSE_train, x_axis, SSE_val)
plt.title("SSE for training and validation data, alpha = " + str(alpha))
plt.xlabel("training iteration")
plt.ylabel("SSE")
plt.legend(["SSE_Training", "SSE_Testing"])
plt.show()

plt.figure(1)

plt.plot(x_axis, w_grad_norms)
plt.title("Norm of gradients")
plt.xlabel("training iteration")
plt.ylabel("gradient norm")
plt.show()

