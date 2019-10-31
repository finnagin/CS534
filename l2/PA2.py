import pandas as pd
import zipfile as zf
import numpy as np


def loadzip(zipname, csvname):
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname), header=None)
    df[0][df[0] == 3] = 1
    df[0][df[0] == 5] = -1
    df[df.shape[1]] = [1.]*len(df)
    return df

def online_perceptron(df,iters):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    w = np.array([0]*X.shape[1])
    w_list = []
    for i in range(iters):
        w_list.append(w)
        A = y*np.dot(X,w)
        w = w + sum((np.tile(y,(X.shape[1],1)).T*X)[A <= 0])
    w_list.append(w)
    return w_list

def avg_perceptron(df,iters):
    # still need to tinker with this
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    w = np.array([0]*X.shape[1])
    w_ = np.array([0]*X.shape[1])
    w_list = []
    for i in range(iters):
        w_list.append(w)
        A = y*np.dot(X,w)
        update_vals = (np.tile(y,(X.shape[1],1)).T*X)
        w = w + sum(update_vals[A <= 0])
    w_list.append(w)
    return w_list

def kernel_perceptron(df,iters):
    pass



df = loadzip('data/pa2_train.csv.zip')


