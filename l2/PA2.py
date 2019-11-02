import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse


def loadzip(zipname, csvname, test=False):
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname), header=None)
    if not test:
        df[0][df[0] == 3] = 1
        df[0][df[0] == 5] = -1
    df[df.shape[1]] = [1.]*len(df)
    return df

def online_perceptron_loop(df, iters):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    w = np.zeros(X.shape[1])
    w_list = []
    n = X.shape[0]
    for i in range(iters):
        w_list.append(w.copy())
        for idx in range(n):
            if np.dot(X[idx],w)*y[idx] <= 0:
                w = w + X[idx]*y[idx]
    w_list.append(w.copy())
    return w_list

def avg_perceptron_loop(df, iters):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    n = X.shape[0]
    w = np.array([0]*X.shape[1])
    w_ = np.array([0]*X.shape[1])
    s = 0
    w_list = []
    for i in range(iters):
        w_list.append(w_.copy())
        for idx in range(n):
            if np.dot(X[idx],w)*y[idx] <= 0:
                w = w + X[idx]*y[idx]
            w_ = ((s*w_) + w)/(s+1)
            s += 1
    w_list.append(w_.copy())
    return w_list

def kernel_perceptron_loop(df, iters, p):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    n = X.shape[0]
    a = np.zeros(n)
    K = (1+np.matmul(X,X.T))**p
    a_list = []
    for i in range(iters):
        a_list.append(a.copy())
        for idx in range(n):
            u = sum(a*K[idx]*y)
            if u*y[idx] <= 0:
                a[idx] += 1
    a_list.append(a.copy())
    return a_list

def KernelizedPerceptron(Y_train, X_train, p, iters):
    (num_of_examples, feature_number) = X_train.shape
    a = np.zeros(num_of_examples)
    weight_list = []
    weight_list.append(np.copy(a))
    K = np.power(1 + np.matmul(X_train, np.transpose(X_train)),p)
    iter = 0
    while iter < iters:
        for t in range(num_of_examples):
            u = np.dot(a, np.transpose(K[:,t])*Y_train)
            if Y_train[t]*u <= 0:
                a[t] += 1
        weight_list.append(np.copy(a))
        iter += 1
    return weight_list

def predictK(a, K, y, X_test):
    preds = np.sign(np.dot(K,a*y))
    return preds


"""
def online_perceptron(df,iters):
    # this is broken as it needs to update w every step not atthe end
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    w = np.zeros(X.shape[1])
    w_list = []
    for i in range(iters):
        w_list.append(w.copy())
        A = y*np.dot(X,w)
        w = w + sum((np.tile(y,(X.shape[1],1)).T*X)[A <= 0])
    w_list.append(w.copy())
    return w_list

def avg_perceptron(df,iters):
    # This is incorect. Not sure How to apply vectorization correctly.
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    n = X.shape[0]
    w = np.array([0]*X.shape[1])
    w_ = np.array([0]*X.shape[1])
    s = 0
    w_list = []
    for i in range(iters):
        s0 = s.copy()
        w_list.append(w.copy())
        A = y*np.dot(X,w)
        update_vals = (np.tile(y,(X.shape[1],1)).T*X)
        w = w + sum(update_vals[A <= 0])
        s1 = s + update_vals[A <= 0].shape[0]
        s2 = s + n - s
        if s > 0:
            w_ = (1/s)*(s0*w_ + w)
    w_list.append(w.copy())
    return w_list

def kernel_perceptron(df,iters, p):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    a = np.zeros(X.shape[0])
    K = (1+np.matmul(X,X.T))**p
    a_list = []
    for i in range(iters):
        a_list.append(a.copy())
        u = np.dot(a,(K*y))
        u1 = u*y
        a = a + (u1 <= 0)
    a_list.append(a.copy())
    return a_list
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
    parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

    args = parser.parse_args()
    if True:
        df = loadzip('data/pa2_train.csv.zip','pa2_train.csv')
        df_val = loadzip('data/pa2_valid.csv.zip','pa2_valid.csv')
        df_test = loadzip('data/pa2_test_no_label.csv.zip','pa2_test_no_label.csv',True)
        X = np.array(df.values[:,1:])
        y = np.array(df[0].values)
        n = X.shape[0]
        X_val = np.array(df_val.values[:,1:])
        y_val = np.array(df_val[0].values)
        X_test = np.array(df_test.values)
        n_test = X_test.shape[0]
        n_val = X_val.shape[0]
        iters = 15

        
    if 1 in args.parts:
        print("Starting part 1...")
        w1s = online_perceptron_loop(df, iters)
        train1 = []
        val1 = []
        for w in w1s:
            tmp_train = 0
            tmp_val = 0
            for idx in range(n):
                tmp_train += int(np.dot(X[idx],w)*y[idx] <= 0)
            train1.append(tmp_train)
            for idx in range(n_val):
                tmp_val += int(np.dot(X_val[idx],w)*y_val[idx] <= 0)
            val1.append(tmp_val)
        best_iter = val1.index(min(val1))
        with open("oplabel.csv","w") as fid:
            pred1 = np.sign(np.dot(X_test,w1s[best_iter]))
            for idx in range(n_test):
                fid.write(str(pred1[idx]))
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
        print("Starting part 2...")
        w2s = avg_perceptron_loop(df, iters)
        train2 = []
        val2 = []
        for w in w2s:
            tmp_train = 0
            tmp_val = 0
            for idx in range(n):
                tmp_train += int(np.dot(X[idx],w)*y[idx] <= 0)
            train2.append(tmp_train)
            for idx in range(n_val):
                tmp_val += int(np.dot(X_val[idx],w)*y_val[idx] <= 0)
            val2.append(tmp_val)
        best_iter = val2.index(min(val2))
        with open("aplabel.csv","w") as fid:
            pred2 = np.sign(np.dot(X_test,w2s[best_iter]))
            for idx in range(n_test):
                fid.write(str(pred2[idx]))
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
        all_a = []
        Ks = []
        K_vals = []
        for p in [1,2,3,4,5]:
            K = (1+np.matmul(X_val,X.T))**p
            K_vals.append(K)
            K = (1+np.matmul(X,X.T))**p
            Ks.append(K)
            all_a.append(KernelizedPerceptron(y, X, p, iters))
            #all_a.append(kernel_perceptron_loop(df,iters,p))

    if 3 in args.parts:
        trains = []
        vals = []
        for p in range(len(all_a)):
            train=[]
            val=[]
            for idx in range(len(all_a[p])):
                print("a"+str(idx))
                preds = predictK(all_a[p][idx], Ks[p], y, X)
                train.append(sum(preds*y<=0))
                preds = predictK(all_a[p][idx], K_vals[p], y, X_val)
                val.append(sum(preds*y_val<=0))
            trains.append(train)
            vals.append(val)

    if 3 in args.parts:
        p = 1
        best_iters = [val.index(min([x for x in val])) for val in vals]
        min_errs = [min([x for x in val]) for val in vals]
        best_p = min_errs.index(min(min_errs))
        best_iter = best_iters[best_p]
        K_test = (1+np.matmul(X_test,X.T))**(best_p+1)
        preds = predictK(all_a[best_p][best_iter], K_test, y, X_test)
        with open("kplabel.csv","w") as fid:
            for idx in range(n_test):
                fid.write(str(preds[idx]))
        if not args.hide:
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
            plt.figure()
            plt.plot([1,2,3,4,5],[1-x/float(n_val) for x in min_errs],color="#984ea3")
            lt.title("Best Validation Accuracy for P Values")
            plt.xlabel("P")
            plt.ylabel("Accuracy")
            plt.show()


