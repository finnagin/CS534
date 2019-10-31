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
    # this is broken as it needs to update w t every step not atthe end
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    w = np.zeros(X.shape[1])
    w_list = []
    for i in range(iters):
        w_list.append(w)
        A = y*np.dot(X,w)
        w = w + sum((np.tile(y,(X.shape[1],1)).T*X)[A <= 0])
    w_list.append(w)
    return w_list

def avg_perceptron(df,iters):
    # This is incorect. Noit sure How to apply vectorization correctly.
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    n = X.shape[0]
    w = np.array([0]*X.shape[1])
    w_ = np.array([0]*X.shape[1])
    s = 0
    w_list = []
    for i in range(iters):
        s0 = s.copy()
        w_list.append(w)
        A = y*np.dot(X,w)
        update_vals = (np.tile(y,(X.shape[1],1)).T*X)
        w = w + sum(update_vals[A <= 0])
        s1 = s + update_vals[A <= 0].shape[0]
        s2 = s + n - s
        if s > 0:
            w_ = (1/s)*(s0*w_ + w)
    w_list.append(w)
    return w_list

def kernel_perceptron(df,iters, p):
    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    a = np.zeros(X.shape[0])
    K = (1+np.matmul(X,X.T))**p
    a_list = []
    for i in range(iters):
        a_list.append(a)
        u = np.dot(a,(K*y))
        a = a + u[u <= 0]
    a_list.append(a)
    return a_list



df = loadzip('data/pa2_train.csv.zip')


