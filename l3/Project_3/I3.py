# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:00:58 2019

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import Decision_Tree_Node as DTN
import Random_Forest_Module as RFM
import Ada_Boosted_Decision_Tree_Module as ABDTM

# load CSV file and create feature dictionary.

args = "3"

with open("pa3_train.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader)
    
    training_data_list = list(csvreader)
    
    training_data = np.array(training_data_list).astype(float)
    
    
with open("pa3_val.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader)
    
    val_data_list = list(csvreader)   
    val_data = np.array(val_data_list).astype(float)
    



feature_dictionary = {};

for i in range(len(header)-1):
    feature_dictionary[header[i]] = i



training_data_shape = training_data.shape;
val_data_shape = val_data.shape

if "1" in args:
    
    print()
    print("Decision tree for depth in [0,1,2,3,4,5,6,7,8]")
    print()

    training_accuracy = []
    val_accuracy = []
    
    weights = np.ones(training_data_shape[0])

    
    for max_depth in range(9):

        root_node = DTN.Decision_Tree_Node(feature_dictionary, training_data, max_depth, 0, weights, None)
        
        Correct_Count = 0;

        for index in range(training_data_shape[0]):
            
            if root_node.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1

    
        training_accuracy.append(Correct_Count/training_data_shape[0])

        print("Training data accuracy for depth " + str(max_depth) + " is: " + str(training_accuracy[max_depth]))
    
    
        
        Correct_Count = 0;
   
        for index in range(val_data_shape[0]):
            
            if root_node.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1
    
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for depth " + str(max_depth) + " is: " + str(val_accuracy[max_depth]))
        
        
        
    plt.plot(np.linspace(0, 8, 9), training_accuracy,'rs-', np.linspace(0,8,9), val_accuracy, 'b^-')
    plt.legend(("Training", "Validation"))
    plt.ylabel("accuracy")
    plt.xlabel("tree depth")
    plt.show()
        
if "2" in args:
    
    # loop over n, tree number
    
    max_depth = 2
    m = 5
    
    weights = np.ones(training_data_shape[0])
    
    training_data_shape = training_data.shape;
    val_data_shape = val_data.shape
    weights = np.ones(training_data_shape[0])
    
    print()
    print("Random Forest with d = 2, m = 5 and n in [1,2,5,10,25]")
    print()
    
    training_accuracy = []
    val_accuracy = []
    
    
    n_array = [1,2,5,10,25]
    
    for n in n_array:
    
        forest = RFM.Random_Forest(feature_dictionary, training_data, n, m, max_depth, weights)
        
        Correct_Count = 0

        for index in range(training_data_shape[0]):

            if forest.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1
                
        training_accuracy.append(Correct_Count/training_data_shape[0])

        print("Training data accuracy for n: " + str(n) + " is: " + str(training_accuracy[-1]))
        
    
        Correct_Count = 0
   
        for index in range(val_data_shape[0]):
        
            if forest.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
                
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for n: " + str(n) + " is: " + str(val_accuracy[-1]))
    
    plt.plot(n_array, training_accuracy,'rs-', n_array, val_accuracy, 'b^-')
    plt.legend(("Training", "Validation"))
    plt.ylabel("accuracy")
    plt.xlabel("tree number")
    plt.show()
    
    print("Best n value is:" + str(n_array[np.argmax(val_accuracy)]))
    
    # loop over m, feature sample number
    
    
    max_depth = 2
    n = 15
    
    training_data_shape = training_data.shape;
    val_data_shape = val_data.shape
    weights = np.ones(training_data_shape[0])
    
    print()
    print("Random Forest with d = 2, n = 15 and m in [1,2,5,10,25,50]")
    print()
    
    training_accuracy = []
    val_accuracy = []
    
    
    m_array = [1,2,5,10,25,50]
    
    for m in m_array:
    
        forest = RFM.Random_Forest(feature_dictionary, training_data, n, m, max_depth, weights)
        
        Correct_Count = 0

        for index in range(training_data_shape[0]):

            if forest.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1
                
        training_accuracy.append(Correct_Count/training_data_shape[0])

        print("Training data accuracy for m: " + str(m) + " is: " + str(training_accuracy[-1]))
        
    
        Correct_Count = 0
   
        for index in range(val_data_shape[0]):
        
            if forest.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
                
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for m: " + str(m) + " is: " + str(val_accuracy[-1]))
    
    plt.plot(m_array, training_accuracy,'rs-', m_array, val_accuracy, 'b^-')
    plt.legend(("Training", "Validation"))
    plt.ylabel("accuracy")
    plt.xlabel("feature sample number")
    plt.show()
    
    print("Best m value is:" + str(m_array[np.argmax(val_accuracy)]))
    
    
    # for n = 10, m = 50, loop 10 times and average result
    
    
    
    max_depth = 2
    n = 10
    m = 50
    
    training_data_shape = training_data.shape;
    val_data_shape = val_data.shape
    weights = np.ones(training_data_shape[0])
    
    max_trials = 10
    
    print()
    print("With n: " + str(n) + " and m: " +str(m) + " and "+str(max_trials)+" trials")
    print()
    
    training_accuracy = []
    val_accuracy = []
    
    
    for t in range(max_trials):
    
        forest = RFM.Random_Forest(feature_dictionary, training_data, n, m, max_depth, weights)
        
        Correct_Count = 0

        for index in range(training_data_shape[0]):

            if forest.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1
                
        training_accuracy.append(Correct_Count/training_data_shape[0])

        print("Training data accuracy for trial: " + str(t) + " is: " + str(training_accuracy[-1]))
        
    
        Correct_Count = 0
   
        for index in range(val_data_shape[0]):
        
            if forest.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
                
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for trial: " + str(t) + " is: " + str(val_accuracy[-1]))
    
#    plt.plot(np.linspace(1,max_trials,max_trials), training_accuracy,'rs', np.linspace(1,max_trials,max_trials), val_accuracy, 'b^')
#    plt.legend(("Training", "Validation"))
#    plt.ylabel("accuracy")
#    plt.xlabel("trial")
#    plt.show()
    
    train_average_accuracy = sum(training_accuracy)/len(training_accuracy)
    val_average_accuracy = sum(val_accuracy)/len(val_accuracy)
    
    print()
    print("Average training accuracy:"+str(train_average_accuracy))
    print("Average validation accuracy:"+str(val_average_accuracy))
    
    
if "3" in args:
    
    max_depth = 1   
    training_data_shape = training_data.shape;
    val_data_shape = val_data.shape
    
    training_accuracy = []
    val_accuracy = []
    
    
    Boosted_Learner_List = [] 

    L_values = [1,2,5,10,15]
    
    print()
    print("Looping over L values: " + str(L_values))
    print()
    
    for L in L_values:
        
        Boosted_Learner = ABDTM.Ada_Boosted_Decision_Tree(feature_dictionary, training_data, max_depth, L)
        
        Correct_Count = 0
        
        for index in range(training_data_shape[0]):
            
            if Boosted_Learner.make_prediction(training_data[index,:], False) == training_data[index, training_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
            
        training_accuracy.append(Correct_Count/training_data_shape[0])
            
        print("Training data accuracy for L: " + str(L) + " is: " + str(training_accuracy[-1]))
            
        Correct_Count = 0
        
        for index in range(val_data_shape[0]):
             
            if Boosted_Learner.make_prediction(val_data[index,:],False) == val_data[index, val_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
            
        val_accuracy.append(Correct_Count/val_data_shape[0])
        
        print("Validation data accuracy for L: " + str(L) + " is: " + str(val_accuracy[-1]))
        
    
    plt.plot(L_values, training_accuracy,'rs-', L_values, val_accuracy, 'b^-')
    plt.legend(("Training", "Validation"))
    plt.ylabel("accuracy")
    plt.xlabel("L")
    plt.show()
            
            
        
                
        
    
    
    
    
    
    
    
    
    
        
     


    
    




          
         
         

