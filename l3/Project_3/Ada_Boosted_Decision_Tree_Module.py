# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:48:19 2019

@author: Michael
"""

import numpy as np
import Decision_Tree_Module as DTM

class Ada_Boosted_Decision_Tree:
    
    def __init__(self, features, training_samples, max_depth, L):
        
        self.training_samples_shape = training_samples.shape
        self.L = L
        self.h_list = []
        self.D_list = []
        self.alpha_list = [0]*L        
        self.error_list = [0]*L
        
        # initialize weight to place equal weight on all samples
        
        self.D_list.append(np.ones(self.training_samples_shape[0])/self.training_samples_shape[0])
        
        for l in range(self.L):
            
            self.h_list.append(DTM.Decision_Tree_Node(features, training_samples, max_depth, 0, self.D_list[l], None))      
            self.error_list[l] = 0
            
            # compute weighted error
            
            for i in range(self.training_samples_shape[0]):
                
                if self.h_list[l].make_prediction(training_samples[i,:], False) != training_samples[i,self.training_samples_shape[1]-1]:
                    
                    self.error_list[l] = self.error_list[l] + self.D_list[l][i]
                    
            # compute alpha
                
            self.alpha_list[l] = 1/2*np.log((1-self.error_list[l])/self.error_list[l]) 
            
            
            # compute next iterations weights
            
            self.D_list.append(np.copy(self.D_list[l]))
            
            for i in range(self.training_samples_shape[0]): 
                
                
                if self.h_list[l].make_prediction(training_samples[i,:], False) != training_samples[i,self.training_samples_shape[1]-1]:
                    self.D_list[l+1][i] = np.exp(self.alpha_list[l])*self.D_list[l][i]
                else:
                    self.D_list[l+1][i] = np.exp(-1*self.alpha_list[l])*self.D_list[l][i]
                    
            # normalize next weights to 1.
            
            self.D_list[l+1] = np.copy(self.D_list[l+1])/np.sum(self.D_list[l+1])
            
    
                    
            
    def make_prediction(self, sample, print_error):
        
        weighted_weak_learner_predictions = []
        
        
        for l in range(self.L):
            
            prediction = self.h_list[l].make_prediction(sample, print_error)
            
            if prediction == 0:
                prediction = -1
            else:
                prediction = 1
            
            weighted_weak_learner_predictions.append(prediction*self.alpha_list[l])
            
        if(sum(weighted_weak_learner_predictions) >= 0):
            return 1
        else:
            return 0
            
        