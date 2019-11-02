# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:57:27 2019

@author: Michael
"""

args = "0 1 2"

import numpy as np
import matplotlib.pyplot as plt
from Perceptron_Algorithms import *

# %%

# Part 0 data loading

if "0 1" in args:

    data_train = np.loadtxt('pa2_train.csv', delimiter=',')


    Y_train = data_train[:,0]
    X_train = np.ones((Y_train.size,1))
    
    X_train = np.append(X_train, data_train[:,1:], axis=1)

    data_val = np.loadtxt('pa2_valid.csv', delimiter=',')

    Y_val = data_val[:,0]
    X_val = np.ones((Y_val.size,1))
    
    X_val = np.append(X_val, data_val[:,1:], axis=1)

    
    data_test = np.loadtxt('pa2_test_no_label.csv',delimiter=',')
      
    (test_num, feature_num) = np.shape(data_test)
    
    X_test = np.zeros((test_num, 1))
    
    X_test = np.append(X_test, data_test, axis=1)
    


#image = np.reshape(X_train[0,:], (28,28))

#plt.imshow(image)


    np.place(Y_train, Y_train == 3, 1)
    np.place(Y_train, Y_train == 5, -1)
    
    np.place(Y_val, Y_val == 3, 1)
    np.place(Y_val, Y_val == 5, -1)

    #(num_of_examples, feature_number) = X_train.shape

    iters = 15
 
# %%

# Part 1

if "1" in args:

    weight_list = OnlinePerceptron(Y_train, X_train, iters)

    train_errors_per_iteration = []
    val_errors_per_iteration = []


    for W_index in range(len(weight_list)):
       
        train_prediction = np.sign(np.dot(X_train, weight_list[W_index]))
        train_errors = np.count_nonzero(train_prediction - Y_train)                                     
        
        val_prediction = np.sign(np.dot(X_val,weight_list[W_index]))
        val_errors = np.count_nonzero(val_prediction-Y_val)
                                                   
        train_errors_per_iteration.append(train_errors)
        val_errors_per_iteration.append(val_errors)
            
    
    plt.plot(range(len(weight_list)),train_errors_per_iteration, label='training error')
    plt.plot(range(len(weight_list)), val_errors_per_iteration, label='validaiton error')
    plt.title('Perceptron Algorithm')
    plt.ylabel('number of errors')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()
    
    best_weight_index = np.argmin(val_errors_per_iteration)
    
    best_weight = weight_list[best_weight_index]
    
    print("The best weight is at iteration: "+ str(best_weight_index) + " with validation error: " + str(val_errors_per_iteration[best_weight_index]))
    test_prediction = np.sign(np.dot(X_test, best_weight))
    
    np.savetxt('oplabel.csv',test_prediction, delimiter = ',')
       

# %%

# Part 2

if "2" in args:

    weight_list = AveragePerceptron(Y_train, X_train, iters)

    train_errors_per_iteration = []
    val_errors_per_iteration = []


    for W_index in range(len(weight_list)):
        
        
        train_prediction = np.sign(np.dot(X_train, weight_list[W_index]))
        train_errors = np.count_nonzero(train_prediction - Y_train)                                     
        
        val_prediction = np.sign(np.dot(X_val,weight_list[W_index]))
        val_errors = np.count_nonzero(val_prediction-Y_val)
                                                   
        train_errors_per_iteration.append(train_errors)
        val_errors_per_iteration.append(val_errors)

    
    plt.figure()
    plt.plot(range(len(weight_list)),train_errors_per_iteration, label='training error')
    plt.plot(range(len(weight_list)), val_errors_per_iteration, label='validaiton error')
    plt.title('Average Perceptron Algorithm')
    plt.ylabel('number of errors')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()
    best_weight_index = np.argmin(val_errors_per_iteration)
    
    best_weight = weight_list[best_weight_index]
    
    print("The best weight is at iteration: "+ str(best_weight_index) + " with validation error: " + str(val_errors_per_iteration[best_weight_index]))
    test_prediction = np.sign(np.dot(X_test, best_weight))
    
    np.savetxt('oplabel.csv',test_prediction, delimiter = ',')

     
# %%

# Part 3

if "3" in args:
    
    
    for p in [1,2,3,4,5]:
        
        weight_list = KernelizedPerceptron(Y_train, X_train, p, iters)

        train_errors_per_iteration = []
        val_errors_per_iteration = []


        for W_index in range(len(weight_list)):
    
            train_prediction = KernelizedPerceptronPrediction(weight_list[W_index], X_train, Y_train, X_train, p)      
            train_errors = np.count_nonzero(train_prediction - Y_train)
       
            val_prediction = KernelizedPerceptronPrediction(weight_list[W_index], X_val, Y_train, X_train, p)      
            val_errors = np.count_nonzero(val_prediction - Y_val)
       
            train_errors_per_iteration.append(train_errors)
            val_errors_per_iteration.append(val_errors)
       
        plt.figure()
        plt.plot(range(len(train_errors_per_iteration)),train_errors_per_iteration, label='training error')
        plt.plot(range(len(val_errors_per_iteration)), val_errors_per_iteration, label='validaiton error')
        plt.title('Kernel Perceptron Algorithm, p: ' + str(p))
        plt.ylabel('number of errors')
        plt.xlabel('iterations')
        plt.legend()
        
        

    
       


       