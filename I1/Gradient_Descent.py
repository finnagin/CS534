# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:26:22 2019

@author: Michael
"""

import numpy as np 

# this class runs gradient descent for linear regression, it accepts a target training vector along with a training features matrix where each row represents a single example,
# additionally it requires a learning rate, regularization constant and two stopping criteria, norm on the gradient and a maximum number of iterations
# which ever is reached first terminates the descent, addid more stuff

class Gradient_Descent:
    
    def run_gradient_descent(self,target_training_vector, training_features_matrix, starting_weights, learning_rate, regularization_constant, gradient_norm_stopping_criteria, max_iterations):
        
        number_of_training_examples = target_training_vector.size
        number_of_features = starting_weights.size
        
        weight_vector = starting_weights
        weight_vector_gradient = np.zeros(number_of_features)
        
        
        training_iteration = 0 
        
        
        # calculuate sum of squared errors for initial starting weight
        
        training_vector_index = 0
        Sum_of_Squared_Errors = []
        norm_of_weight_vector_gradient = []
        
    
        # apply gradient descent until max iteration or stopping criteria is met
        
        while(training_iteration < max_iterations and np.linalg.norm(weight_vector_gradient) >= gradient_norm_stopping_criteria): 
            
            # calculate Sum of Squares Errors for current weights  
        
            Sum_of_Squared_Errors_Current_Weight = 0
            training_vector_index = 0
            
            # using fact python has starting index of 0
            
            while(training_vector_index < number_of_training_examples):
                Sum_of_Squared_Errors_Current_Weight = Sum_of_Squared_Errors_Current_Weight + (target_training_vector[training_vector_index]-np.dot(weight_vector, training_features_matrix[training_vector_index,:]))^2
                training_vector_index = training_vector_index + 1
            
            # append to list of sum of squared errors 
            
            Sum_of_Squared_Errors.append(Sum_of_Squared_Errors_Current_Weight)
            
            
           # compute gradient by computing partial derivative of each weight and updating weight_vector_gradient
         
            weight_index = 0;
        
            # use the fact python has starting index of 0
            
            while(weight_index < number_of_features):
            
                training_vector_index = 0;
                
                # use fact python has starting index of 0
        
                while(training_vector_index < number_of_training_examples):
                    
                    # apply partial derivative formula summing up across all training examples
                    
                    weight_vector_gradient[weight_index] = weight_vector_gradient[weight_index] + 2*(target_training_vector[training_vector_index]- np.dot(weight_vector, training_features_matrix[training_vector_index,:]))*training_features_matrix[training_vector_index,weight_index]
                    training_vector_index = training_vector_index + 1
                    
                # apply weight regularization affect to partial derivative 
                
                weight_vector_gradient[weight_index] = weight_vector_gradient[weight_index] + regularization_constant*2* weight_vector[weight_index]
                
                weight_index = weight_index + 1
        
            # update append norm of current weight vector gradient to list of norms, 
            
            norm_of_weight_vector_gradient.append(np.linalg.norm(weight_vector_gradient))
            
            # update weight vector using learning rate 
        
            weight_vector = weight_vector - learning_rate*weight_vector_gradient 
            
            
        # I wasnt sure if the final SSE was to be computed after the final gradient update, if it is computed the length of the norm_of_weight_vector_gradient will be one less than the length of the Sum_of_Squared_Errors     
            
        """   
        # calculate Sum of Squared Errors for final weight 
            
        Sum_of_Squared_Errors_Current_Weight = 0
        
        training_vector_index = 0
        
        while(training_vector_index < number_of_training_examples):
            
            Sum_of_Squared_Errors_Current_Weight = Sum_of_Squared_Errors_Current_Weight + (target_training_vector[training_vector_index]-np.dot(weight_vector, training_features_matrix[training_vector_index,:]))^2
            training_vector_index = training_vector_index + 1
            
            # append to list of sum of squared errors 
            
            Sum_of_Squared_Errors.append(Sum_of_Squared_Errors_Current_Weight)
        """   
    
        return [weight_vector, Sum_of_Squared_Errors, norm_of_weight_vector_gradient]



    
    
    

    
