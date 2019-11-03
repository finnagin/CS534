import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Perceptron_Algorithms import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
    parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

    args = parser.parse_args()

    # Load the data from the zips
    df = loadzip('data/pa2_train.csv.zip','pa2_train.csv')
    df_val = loadzip('data/pa2_valid.csv.zip','pa2_valid.csv')
    df_test = loadzip('data/pa2_test_no_label.csv.zip','pa2_test_no_label.csv',True)
    
    # Extract the values from the dataframe
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    n = X.shape[0]
    X_val = np.array(df_val.values[:,1:])
    y_val = np.array(df_val[0].values)
    n_val = X_val.shape[0]
    X_test = np.array(df_test.values)
    n_test = X_test.shape[0]
    iters = 15

        
    if 1 in args.parts:
        print("Starting part 1: online perceptron...")
        # get weight vectors
        w1s = online_perceptron_loop(y, X, iters)
        train1 = []
        val1 = []
        for w in w1s:
            tmp_train = 0
            tmp_val = 0
            # get error rates for the training data
            for idx in range(n):
                tmp_train += int(np.dot(X[idx],w)*y[idx] <= 0)
            train1.append(tmp_train)
            # get error rates for the validation data
            for idx in range(n_val):
                tmp_val += int(np.dot(X_val[idx],w)*y_val[idx] <= 0)
            val1.append(tmp_val)
        # get the best iteration from validation data
        best_iter = val1.index(min(val1))
        print("Best Validation Accuracy was with " + str(best_iter) + " iterations: " + str(1-min(val1)/float(n_val)) + "%")
        # predict test values
        with open("oplabel.csv","w") as fid:
            pred1 = np.sign(np.dot(X_test,w1s[best_iter]))
            for idx in range(n_test):
                fid.write(str(int(pred1[idx]))+"\n")
        # plot the figure
        if not args.hide:
            plt.figure()
            plt.plot(range(len(w1s)),[1-x/float(n) for x in train1],color="#ff7f00", label="Train")
            plt.plot(range(len(w1s)),[1-x/float(n_val) for x in val1],color="#984ea3", label="Validation")
            plt.title("Online Perceptron")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    if 2 in args.parts:
        print("Starting part 2: average perceptron...")
        # get weight vectors
        w2s = avg_perceptron_loop(y, X, iters)
        train2 = []
        val2 = []
        for w in w2s:
            tmp_train = 0
            tmp_val = 0
            # count errors for training data
            for idx in range(n):
                tmp_train += int(np.dot(X[idx],w)*y[idx] <= 0)
            train2.append(tmp_train)
            # count errors for validation data
            for idx in range(n_val):
                tmp_val += int(np.dot(X_val[idx],w)*y_val[idx] <= 0)
            val2.append(tmp_val)
        best_iter = val2.index(min(val2))
        print("Best Validation Accuracy was with " + str(best_iter) + " iterations: " + str(1-min(val2)/float(n_val)) + "%")
        # predict test values
        with open("aplabel.csv","w") as fid:
            pred2 = np.sign(np.dot(X_test,w2s[best_iter]))
            for idx in range(n_test):
                fid.write(str(int(pred2[idx]))+"\n")
        # plot the figure
        if not args.hide:
            plt.figure()
            plt.plot(range(len(w2s)),[1-x/float(n) for x in train2],color="#ff7f00", label="Train")
            plt.plot(range(len(w2s)),[1-x/float(n_val) for x in val2],color="#984ea3", label="Validation")
            plt.title("Average Perceptron")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()


    if 3 in args.parts:
        print("Starting part 3: kernel perceptron...")
        all_a = []
        Ks = []
        K_vals = []
        for p in [1,2,3,4,5]:
            # calculate the training Gram matrix
            K = (1+np.matmul(X_val,X.T))**p
            K_vals.append(K)
            # validation Gram Matrix
            K = (1+np.matmul(X,X.T))**p
            Ks.append(K)
            # calculate alpha vectors
            all_a.append(KernelizedPerceptron(y, X, p, iters))
            # depriciated
            #all_a.append(kernel_perceptron_loop(df,iters,p))

    if 3 in args.parts:
        trains = []
        vals = []
        for p in range(len(all_a)):
            train=[]
            val=[]
            for idx in range(len(all_a[p])):
                # count training errors
                preds = predictK(all_a[p][idx], Ks[p], y, X)
                train.append(sum(preds*y<=0))
                # count validation errors
                preds = predictK(all_a[p][idx], K_vals[p], y, X_val)
                val.append(sum(preds*y_val<=0))
            trains.append(train)
            vals.append(val)

    if 3 in args.parts:
        p = 1
        # Get best p and best iteration using Validation error
        best_iters = [val.index(min([x for x in val])) for val in vals]
        min_errs = [min([x for x in val]) for val in vals]
        best_p = min_errs.index(min(min_errs))
        best_iter = best_iters[best_p]
        # Calculate Test Gram Matrix
        K_test = (1+np.matmul(X_test,X.T))**(best_p+1)
        # Make and write test predictions
        preds = predictK(all_a[best_p][best_iter], K_test, y, X_test)
        with open("kplabel.csv","w") as fid:
            for idx in range(n_test):
                fid.write(str(int(preds[idx]))+"\n")
        # Print Validation Accuracy for each p
        for acc in [1-x/float(n_val) for x in min_errs]:
            print("Best Validation Accuracy for p="+str(p)+" is: "+str(acc*100)+"%")
            p+=1
        print("The best value for P is "+str(best_p+1))
        if not args.hide:
            p=1
            # plot for each p
            for train, val in zip(trains, vals):
                plt.figure()
                plt.plot(range(len(train)),[1-x/float(n) for x in train],color="#ff7f00", label="Train")
                plt.plot(range(len(val)),[1-x/float(n_val) for x in val],color="#984ea3", label="Validation")
                plt.legend()
                plt.title("Kernel Perceptron with p="+str(p))
                plt.xlabel("Iterations")
                plt.ylabel("Accuracy")
                plt.show()
                p+=1
            # plot the validation accuracy and p value plot
            plt.figure()
            plt.plot([1,2,3,4,5],[1-x/float(n_val) for x in min_errs],color="#984ea3")
            plt.title("Best Validation Accuracy for P Values")
            plt.xticks([1,2,3,4,5],[1,2,3,4,5])
            plt.xlabel("P")
            plt.ylabel("Accuracy")
            plt.show()


