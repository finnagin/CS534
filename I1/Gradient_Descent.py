# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:26:22 2019

@author: Michael
"""

import numpy as np 

# this class runs gradient descent for linear regression, it accepts a target training vector along with a training features matrix where each row represents a single example,
# additionally it requires a learning rate, regularization constant and two stopping criteria, norm on the gradient and a maximum number of iterations
# which ever is reached first terminates the descent,

class Gradient_Descent:
    
    def run_gradient_descent(self, y, X, w_0, alpha, lambda_0, epsilon, max_iter):
        
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
            
            w_grad = np.zeros(number_of_features)
            
            
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
        
            # append current gradient vector to list
            
            w_grad_vecs.append(w_grad)
            
            # append norm of gradient to array
            
            w_grad_norms.append(np.linalg.norm(w_grad))
            
            # exit loop if done
            
            if(training_iteration > max_iter or np.linalg.norm(w_grad) < epsilon):
                
                return [w_vecs, w_grad_vecs, w_grad_norms]
            
            
            # update weight vector using learning rate 
        
            w = w - alpha*w_grad 
            
            # append new weight vector to weight vector list
            
            w_vecs.append(w)
            
            training_iteration = training_iteration + 1
            


    
    
    

    
