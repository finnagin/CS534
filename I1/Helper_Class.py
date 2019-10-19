  
        # -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:26:22 2019

@author: Michael
"""

import numpy as np 

# this class provides helper functions for implementation assignment 1. 


class Helper_Class:
    
    # runs gradient descent for linear regression, it accepts a target training vector along with a training features matrix where each row represents a single example,
    # additionally it requires a learning rate, regularization constant and two stopping criteria, norm on the gradient and a maximum number of iterations
    # which ever is reached first terminates the descent and returns a list of the weights as a list of numpy arrays, a list of the gradients as a list of numpy arrays 
    # a list of norms of the gradient
    
    def run_gradient_descent(self, y, X, w_0, alpha, lambda_0, epsilon, max_iter, max_grad):
        
        number_of_training_examples = y.size
        number_of_features = w_0.size
        
        
        training_iteration = 0 
        
        
        # calculuate sum of squared errors for initial starting weight
        
        w_grad_norms = []
        
    
        # apply gradient descent until max iteration or stopping criteria is met
       
        w = w_0
        w_vecs = []
        w_grad_vecs = []
        w_vecs.append(w_0)
        
        while(1): 
            
            
           # compute gradient by computing partial derivative of each weight and updating weight_grad
         
            w_index = 0;
        
            # use the fact python has starting index of 0
            
            #w_grad = np.zeros(number_of_features)
            
            w_grad = sum(((np.dot(X,w)-y)*X.T).T) + lambda_0*2*np.insert(w[1:],0,0)
            #w_grad_test = sum((2*(np.dot(X,w)-y)*X.T).T) + lambda_0*2*np.insert(w[1:],0,0)
            """
            while(w_index < number_of_features):
            
                y_index = 0;
                
                # use fact python has starting index of 0
        
                while(y_index < number_of_training_examples):
                    
                    # apply partial derivative formula summing up across all training examples
                    
                    w_grad[w_index] = w_grad[w_index] + (-1)*2*(y[y_index]- np.dot(w, X[y_index,:]))*X[y_index,w_index]
                    y_index = y_index + 1
                    
                # apply weight regualization effect to partial excluding bias term
                
                if(w_index > 0):
                    w_grad[w_index] = w_grad[w_index] + lambda_0*2* w[w_index]
                
                w_index = w_index + 1
            """
            # append current gradient vector to list
            
            w_grad_vecs.append(w_grad)
            
            # append norm of gradient to array
            
            w_grad_norms.append(np.linalg.norm(w_grad))
            
            # exit loop if done
            
            if(training_iteration > max_iter or np.linalg.norm(w_grad) < epsilon or w_grad_norms[len(w_grad_norms)-1] > max_grad):
                
                return (w_vecs, w_grad_vecs, w_grad_norms)
            
            
            # update weight vector using learning rate 
        
            w = w - alpha*w_grad 
            
            # append new weight vector to weight vector list
            
            w_vecs.append(w)
            
            training_iteration = training_iteration + 1
    
    
    # calculuate the SSE between the target vector y, the feature matrix X where the rows of X are the trainining examples and the columns are the features
    # for each weight vector in the list of numpy arrays representing the weight vectors 
    def calculuate_SSE(self, w_vecs, y, X,):
        
        number_of_w_vecs = len(w_vecs)
        SSE = np.zeros(number_of_w_vecs)
        
        
        # loop over all weight vectors 
        w_vecs_index = 0
        
        while(w_vecs_index < number_of_w_vecs):
        
            y_index = 0
            
            # loop over each target variable, calculating error and adding to total sum
            print(w_vecs[w_vecs_index])
            print(X)
            print(w_vecs[w_vecs_index]*X)
            sse_test = sum(np.square(y-np.dot(X,w_vecs[w_vecs_index])))
            while(y_index < y.size):
                
                
                SSE[w_vecs_index] = SSE[w_vecs_index] + np.square((y[y_index] - np.dot(w_vecs[w_vecs_index],X[y_index,:])))
                y_index = y_index + 1
                
            
            print(SSE[w_vecs_index])
            print(sse_test)
            print(sse_test == SSE[w_vecs_index])
            w_vecs_index = w_vecs_index + 1
        
        return SSE
    
    
    
                
                
        
        
        
            


            

