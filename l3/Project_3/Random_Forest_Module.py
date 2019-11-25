# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:59:38 2019

@author: Michael
"""

import numpy as np
import random as rand
import Random_Forest_Module as RFM

class Random_Forest:
    
    def __init__(self, features, training_data, n, m, max_depth, weights):
        
        self.root_list = [];
        
        self.training_data_shape = training_data.shape;
        
        
        for i in range(n):
            idx = np.random.randint(0,self.training_data_shape[0]-1,size=self.training_data_shape[0])
            
            random_training_sample = training_data[idx,:];
            
            new_root = Random_Forest_Tree_Node(features, m, random_training_sample, max_depth, 0, weights, None)
            
            self.root_list.append(new_root)
        
            
    def make_prediction(self, sample_to_predict, print_tree):
        
        prediction_sum = 0
        
        for root in self.root_list:
            
            if print_tree == True:
                print()
                print("Next Tree:")
            prediction_sum = prediction_sum + root.make_prediction(sample_to_predict, print_tree)
            
        
        if prediction_sum/len(self.root_list) >= 1/2 :
            return 1
        else:
            return 0
            
        
            
        
    

class Random_Forest_Tree_Node:
    
    
    def __init__(self, features_remaining, m, training_data, max_depth, current_depth, weights, parent) :
        
        self.parent = parent
        self.current_depth = current_depth
        self.max_depth = max_depth
        
        self.features_remaining = features_remaining.copy();
        
        self.splitting_feature = None
        
        self.f0_child = None
        self.f1_child = None
        
        
        # compute class probabilities factoring in weights
        
        self.C_0 = 0;
        self.C_1 = 0;
            
        training_data_shape = training_data.shape
            
        for sample in range(training_data_shape[0]):
            if(training_data[sample, training_data_shape[1]-1] == 0):            
                self.C_0 = self.C_0 + weights[sample]
            else:
                self.C_1 = self.C_1 + weights[sample]
            
        self.p_C_0 = self.C_0/(self.C_0 + self.C_1)
        self.p_C_1 = 1 - self.p_C_0
        
        
        if(current_depth < max_depth):
            
        # compute gini uncertainity for node
            
            U_node = 1 - np.power((self.C_0/(self.C_0+self.C_1)),2) - np.power((self.C_1/(self.C_0 + self.C_1)),2)
        
        # select m random features without replacement from remaining features. 
        
            num_of_features_remaining = len(features_remaining)
            
            
        
            selected_features = rand.sample(features_remaining.keys(), min(m,num_of_features_remaining))
            
        # loop through each remaining feature and compute benefit from it, select maximum one with maximum benefit
        
        
            for feature in selected_features:
                
                f0_C_0 = 0
                f0_C_1 = 0
                
                f1_C_0 = 0
                f1_C_1 = 0
                
                    
                for sample in range(training_data_shape[0]):
                
                
                    if(training_data[sample,features_remaining[feature]] == 0):
                        if(training_data[sample,training_data_shape[1]-1] == 0):
                            f0_C_0 = f0_C_0 + weights[sample]
                        else:
                            f0_C_1 = f0_C_1 + weights[sample]
                        
                    else:
                    
                        if(training_data[sample,training_data_shape[1]-1] == 0):
                            f1_C_0 = f1_C_0 + weights[sample]
                        else:
                            f1_C_1 = f1_C_1 + weights[sample]
                            
                        
            
                pf0 = (f0_C_0 + f0_C_1)/(f0_C_0 + f0_C_1 + f1_C_0 + f1_C_1)
                pf1 = 1 - pf0
            
            
                if pf0 != 0:
                    U_node_f0 = 1 - np.power((f0_C_0/(f0_C_0 + f0_C_1)),2) - np.power((f0_C_1/(f0_C_0 + f0_C_1)),2)
                else:
                    U_node_f0 = 0
                
                if pf1 != 0:       
                    U_node_f1 = 1 - np.power((f1_C_0/(f1_C_0 + f1_C_1)),2) - np.power((f1_C_1/(f1_C_0 + f1_C_1)),2)
                else:
                    U_node_f1 = 0
         

            
                Bf = U_node - pf0*U_node_f0 - pf1*U_node_f1 
            
                if self.splitting_feature is None:
                    self.best_benefit = Bf
                    self.splitting_feature = feature
                else:
                    if self.best_benefit <= Bf:
                        self.splitting_feature = feature
                        self.best_benefit = Bf
        
            # split data on best feature
        
            self.f0_training_data_list = []
            self.f0_weights_list = []
            self.f1_training_data_list = []
            self.f1_weights_list = []
        
        
            for sample in range(training_data_shape[0]):
                if training_data[sample, self.features_remaining[self.splitting_feature]] == 0:
                    self.f0_training_data_list.append(training_data[sample,:])
                    self.f0_weights_list.append(weights[sample])
                else:
                    self.f1_training_data_list.append(training_data[sample,:])
                    self.f1_weights_list.append(weights[sample])

                        
            self.f0_training_data = np.array(self.f0_training_data_list)
            self.f0_weights = np.array(self.f0_weights_list)
            self.f1_training_data = np.array(self.f1_training_data_list)
            self.f1_weights = np.array(self.f1_weights_list)
        
            # record feature index and remove from remaining features
                
            self.splitting_feature_index = self.features_remaining[self.splitting_feature]
        
            del self.features_remaining[self.splitting_feature]
           
            # create child nodes, only creating nodes if there are training data
        
        
            if(len(self.f0_weights_list) > 0 ):
                self.f0_child = RFM.Random_Forest_Tree_Node(self.features_remaining, m, self.f0_training_data, self.max_depth, self.current_depth + 1, self.f0_weights, self)
            if(len(self.f1_weights_list) > 0):
                self.f1_child = RFM.Random_Forest_Tree_Node(self.features_remaining, m, self.f1_training_data, self.max_depth, self.current_depth + 1, self.f1_weights, self)
                
    
    
    def make_prediction(self, sample_to_predict, print_tree):
        
        
        if self.splitting_feature is None:
            
            if(self.p_C_0 >= 1/2):
                prediction = 0
            else:
                prediction = 1
                    
            if(print_tree == True):
                print("Terminated at depth: "+ str(self.current_depth))
                print("Prediction is class : " + str(prediction))
                print("Class probabilities, Class 0: " + str(self.p_C_0) + " Class 1: " + str(1-self.p_C_0))
                    
            return prediction
            
            
        
        if(sample_to_predict[self.splitting_feature_index] == 0):
            
            if self.f0_child == None:
                if(self.p_C_0 >= 1/2):
                    prediction = 0
                else:
                    prediction = 1
                    
                if(print_tree == True):
                    print("Terminated at depth: "+ str(self.current_depth))
                    print("Prediction is class : " + str(prediction))
                    print("Class probabilities, Class 0: " + str(self.p_C_0) + " Class 1: " + str(1-self.p_C_0))
                    
                return prediction 
                    
                    
            else:
                if(print_tree == True):
                    print("Current depth: " + str(self.current_depth))
                    print("Branching on feature: "+ str(self.splitting_feature) + " with value " + str(sample_to_predict[self.splitting_feature_index])) 
                
                prediction = self.f0_child.make_prediction(sample_to_predict, print_tree)
                
        else:
                        
            if self.f1_child == None:
                if(self.p_C_0 >= 1/2):
                    prediction = 0
                else:
                    prediction = 1
                    
                if(print_tree == True):
                    print("Terminated at depth: "+ str(self.current_depth))
                    print("Prediction is class : " + str(prediction))
                    print("Class probabilities, Class 0: " + str(self.p_C_0) + " Class 1: " + str(1-self.p_C_0))
                
                return prediction
                    
            else:
                
                if(print_tree == True):
                    print("Current depth: " + str(self.current_depth))
                    print("Branching on feature: "+ str(self.splitting_feature) + " with value " + str(sample_to_predict[self.splitting_feature_index])) 
                
                prediction = self.f1_child.make_prediction(sample_to_predict, print_tree)
                
        return prediction
                