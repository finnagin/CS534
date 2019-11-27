
import numpy as np
import matplotlib.pyplot as plt
import zipfile as zf
#import pandas as pd
import sys
import csv
import Decision_Tree_Module as DTN
import Random_Forest_Module as RFM
import Ada_Boosted_Decision_Tree_Module as ABDTM
import argparse
import time
import random as rand

# load CSV file and create feature dictionary.

# def loadzip(zipname, csvname):
#     """
#     This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

#     :param zipname: A string containing the path/filename for the zip you want to extract from
#     :param csvname: A string containing the filename for the csv file inside the above zip
#     """
#     reader = zf.ZipFile(zipname)
#     df = pd.read_csv(reader.open(csvname))
#     return df

with open("data/pa3_train.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader)
   
    training_data_list = list(csvreader)
   
    training_data = np.array(training_data_list).astype(float)
    
    
with open("data/pa3_val.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader)
  
    val_data_list = list(csvreader)   
    val_data = np.array(val_data_list).astype(float)
    

with open("data/pa3_test.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader)
   
    test_data_list = list(csvreader)   
    test_data = np.array(test_data_list).astype(float)

parser = argparse.ArgumentParser()
parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")
parser.add_argument("--rseed", "-r", action='store_true', help="Add if you want to seed from the time rather than first 8 fibanochi numbers")

args = parser.parse_args()

if args.rseed:
    seed = int(time.time())
else:
    seed = 1123581321
np.random.seed(seed)
rand.seed(seed)



# Load the data from the zips
#df = loadzip('../data/pa3_train.zip','pa3_train.csv')
#training_data = df.values.astype(float)
#header = list(df.keys())
#df_val = loadzip('../data/pa3_val.zip','pa3_val.csv')
#val_data = df_val.values.astype(float)
#df_test = loadzip('../data/pa3_test.zip','pa3_test.csv')
#test_data = df_test.values.astype(float)




feature_dictionary = {};

for i in range(len(header)):
    feature_dictionary[header[i]] = i



training_data_shape = training_data.shape;
val_data_shape = val_data.shape

if 1 in args.parts:
    
    
    weights = np.ones(training_data_shape[0])

    print("Decision tree for depth: 2")

    root_node = DTN.Decision_Tree_Node(feature_dictionary, training_data, 2, 0, weights, None)
        
    Correct_Count = 0;

    for index in range(training_data_shape[0]):
            
        if root_node.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
            Correct_Count = Correct_Count + 1

    
    training_accuracy = (Correct_Count/training_data_shape[0])

    print("Training data accuracy for depth " + str(2) + " is: " + str(training_accuracy))
    

    Correct_Count = 0;
   
    for index in range(val_data_shape[0]):
            
        if root_node.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                
            Correct_Count = Correct_Count + 1
    
    val_accuracy = Correct_Count/val_data_shape[0]

    print("Validation data accuracy for depth " + str(2) + " is: " + str(val_accuracy))

    training_accuracy = []
    val_accuracy = []

    print()
    print("Decision tree for depth in [1,2,3,4,5,6,7,8]")
    print()
    
    for max_depth in range(1,9):

        root_node = DTN.Decision_Tree_Node(feature_dictionary, training_data, max_depth, 0, weights, None)
        
        Correct_Count = 0;

        for index in range(training_data_shape[0]):
            
            if root_node.make_prediction(training_data[index,:],False) == training_data[index,training_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1

    
        training_accuracy.append(Correct_Count/training_data_shape[0])

        print("Training data accuracy for depth " + str(max_depth) + " is: " + str(training_accuracy[-1]))
    
    
        
        Correct_Count = 0;
   
        for index in range(val_data_shape[0]):
            
            if root_node.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                
                Correct_Count = Correct_Count + 1
    
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for depth " + str(max_depth) + " is: " + str(val_accuracy[-1]))
        
        
    if not args.hide:
        plt.figure()
        plt.plot(list(range(1,9)), training_accuracy,'s-',color="#ff7f00", label="Training")
        plt.plot(list(range(1,9)), val_accuracy, '^-',color="#984ea3", label="Validation")
        plt.legend()
        plt.ylabel("accuracy")
        plt.xlabel("tree depth")
        plt.title("Decision Trees")
        plt.savefig("trees.png")
        
if 2 in args.parts:
    
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
    
    if not args.hide:
        plt.figure()
        plt.plot(n_array, training_accuracy,'s-',color="#ff7f00", label="Training")
        plt.plot(n_array, val_accuracy, '^-',color="#984ea3", label="Validation")
        plt.legend(("Training", "Validation"))
        plt.ylabel("accuracy")
        plt.xlabel("tree number")
        plt.title("Random Forest for different n")
        plt.savefig("randon_forest_n.png")
    

    best_n = n_array[np.argmax(val_accuracy)]
    n_acc = np.max(val_accuracy)
    print("Best n value is:" + str(best_n))
    
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
    
    if not args.hide:
        plt.figure()
        plt.plot(m_array, training_accuracy,'s-',color="#ff7f00", label="Training")
        plt.plot(m_array, val_accuracy, '^-',color="#984ea3", label="Validation")
        plt.legend()
        plt.title("Random Forest for different m")
        plt.ylabel("accuracy")
        plt.xlabel("feature sample number")
        plt.savefig("random_forest_m.png")
    
    best_m = m_array[np.argmax(val_accuracy)]
    m_acc = np.max(val_accuracy)
    print("Best m value is:" + str(best_m))
    
    
    # for n = 10, m = 50, loop 10 times and average result
    
    
    
    max_depth = 2
    if n_acc > m_acc:
        n = best_n
        m = 5
    else:
        m = best_m
        n = 15
    
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

        print("Training data accuracy for trial: " + str(t+1) + " is: " + str(training_accuracy[-1]))
        
    
        Correct_Count = 0
   
        for index in range(val_data_shape[0]):
        
            if forest.make_prediction(val_data[index,:],False) == val_data[index,val_data_shape[1]-1]:
                Correct_Count = Correct_Count + 1
                
        val_accuracy.append(Correct_Count/val_data_shape[0])

        print("Validation data accuracy for trial: " + str(t+1) + " is: " + str(val_accuracy[-1]))
    
#    plt.plot(np.linspace(1,max_trials,max_trials), training_accuracy,'rs', np.linspace(1,max_trials,max_trials), val_accuracy, 'b^')
#    plt.legend(("Training", "Validation"))
#    plt.ylabel("accuracy")
#    plt.xlabel("trial")
#    plt.show()
    
    #train_average_accuracy = sum(training_accuracy)/len(training_accuracy)
    #val_average_accuracy = sum(val_accuracy)/len(val_accuracy)
    train_average_accuracy = np.mean(training_accuracy)
    training_accuracy_std = np.std(training_accuracy)
    val_average_accuracy = np.mean(val_accuracy)
    val_accuracy_std = np.std(val_accuracy)
    
    print()
    print("Average training accuracy +- std:"+str(train_average_accuracy)+"+-"+str(training_accuracy_std))
    print("Average validation accuracy +- std:"+str(val_average_accuracy)+"+-"+str(val_accuracy_std))
    
    
if 3 in args.parts:
    
    max_depth = 1   
    training_data_shape = training_data.shape;
    val_data_shape = val_data.shape
    
    training_accuracy = []
    val_accuracy = []
    

    L_values = [1,2,5,10,15]
    
    print()
    print("With d: 1, Looping over L values: " + str(L_values))
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
        
    if not args.hide:
        plt.figure()
        plt.plot(L_values, training_accuracy,'s-',color="#ff7f00", label="Training")
        plt.plot(L_values, val_accuracy, '^-',color="#984ea3", label="Validation")
        plt.legend()
        plt.title("Adaboost for different L")
        plt.ylabel("accuracy")
        plt.xlabel("L")
        plt.savefig("adaboost.png")

    print()
    print("Compairing (L=6,d=2) to (L=15,d=1)")
    print()
    
    max_depth = 2
    L = 6
            
    Boosted_Learner = ABDTM.Ada_Boosted_Decision_Tree(feature_dictionary, training_data, max_depth, L)
        
    Correct_Count = 0
        
    for index in range(training_data_shape[0]):
            
        if Boosted_Learner.make_prediction(training_data[index,:], False) == training_data[index, training_data_shape[1]-1]:
            Correct_Count = Correct_Count + 1
            
    training_accuracy_d2_l6 = Correct_Count/training_data_shape[0]
            
    print("Training data accuracy for L: " + str(L) + " and d: " + str(max_depth) + " is: " + str(training_accuracy_d2_l6))
            
    Correct_Count = 0
        
    for index in range(val_data_shape[0]):
             
        if Boosted_Learner.make_prediction(val_data[index,:],False) == val_data[index, val_data_shape[1]-1]:
            Correct_Count = Correct_Count + 1
            
    val_accuracy_d2_l6 = Correct_Count/val_data_shape[0]
        
    print("Validation data accuracy for L: " + str(L) + " and d: " + str(max_depth)+ " is: " + str(val_accuracy_d2_l6))
            
    
    
    max_depth = 1
    L = 15
            
    Boosted_Learner = ABDTM.Ada_Boosted_Decision_Tree(feature_dictionary, training_data, max_depth, L)
        
    Correct_Count = 0
        
    for index in range(training_data_shape[0]):
            
        if Boosted_Learner.make_prediction(training_data[index,:], False) == training_data[index, training_data_shape[1]-1]:
            Correct_Count = Correct_Count + 1
            
    training_accuracy_d1_l15 = Correct_Count/training_data_shape[0]
            
    print("Training data accuracy for L: " + str(L) + " and d: " + str(max_depth) + " is: " + str(training_accuracy_d1_l15))
            
    Correct_Count = 0
        
    for index in range(val_data_shape[0]):
             
        if Boosted_Learner.make_prediction(val_data[index,:],False) == val_data[index, val_data_shape[1]-1]:
            Correct_Count = Correct_Count + 1
            
    val_accuracy_d1_l15 = Correct_Count/val_data_shape[0]
        
    print("Validation data accuracy for L: " + str(L) + " and d: " + str(max_depth)+ " is: " + str(val_accuracy_d1_l15))
            
    
    if max(val_accuracy) >= val_accuracy_d2_l6:
        
        best_L = L_values[val_accuracy.index(max(val_accuracy))]
        best_d = 1
    else:
        best_L = 6
        best_d = 2
        
    print("The best L value given is: " + str(best_L) +" the best d value given is: " + str(best_d))
    
    Boosted_Learner = ABDTM.Ada_Boosted_Decision_Tree(feature_dictionary, training_data, best_d, best_L)
    
    test_data_shape = test_data.shape
    
    print()
    print("Writing prediction file: pa3_prediction for samples from pa3_test.csv using L: " + str(best_L) + " and d: " + str(best_d))
    
    with open("pa3_prediction.csv", 'w', newline='') as test_output:
        output_writer = csv.writer(test_output, delimiter=',')
        
        for index in range(test_data_shape[0]):
            prediction = Boosted_Learner.make_prediction(test_data[index,:], False)
            output_writer.writerow(str(prediction))
        
    
    
    
        
    
        
    
    
    
    
    
    
    
    
    
        
     


    
    




          
         
         

