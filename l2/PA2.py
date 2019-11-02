import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt


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
            s0 = s
            s += 1
            if np.dot(X[idx],w)*y[idx] <= 0:
                w = w + X[idx]*y[idx]
            w_ = (1/s)*((s0*w_) + w)
    w_list.append(w_.copy())
    return w_list

def kernel_perceptron_loop(df, iters, p):
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
    df = loadzip('data/pa2_train.csv.zip','pa2_train.csv')
    df_val = loadzip('data/pa2_valid.csv.zip','pa2_valid.csv')
    df_test = loadzip('data/pa2_test_no_label.csv.zip','pa2_test_no_label.csv')
    K_vals = []
    iters = 15
    w1s = online_perceptron_loop(df, iters)
    w2s = avg_perceptron_loop(df, iters)
    all_a = []
    Ks = []
    for p in [1,2,3,4,5]:
        K = (1+np.matmul(X_val,X.T))**p
        K_vals.append(K)
        K = (1+np.matmul(X,X.T))**p
        Ks.append(K)
        all_a.append(kernel_perceptron_loop(df,p,iters))

    X = np.array(df.values[:,1:])
    y = np.array(df[0].values)
    X_val = np.array(df_val.values[:,1:])
    y_val = np.array(df_val[0].values)
    X_test = np.array(df_val.values)

    train = []
    val = []
    for idx in range(len(w1s)):
        train.append(sum(np.dot(np.dot(X,w),y) <= 0)/len(y))
        val.append(sum(np.dot(np.dot(X_val,w),y_val) <= 0)/len(y_val))
    plt.figure()
    plt.plot(range(len(w1s)),train,color="#ff7f00")
    plt.plot(range(len(w1s)),val,color="#984ea3")
    plt.show()

    train = []
    val = []
    for idx in range(len(w2s)):
        train.append(sum(np.dot(np.dot(X,w),y) <= 0)/len(y))
        val.append(sum(np.dot(np.dot(X_val,w),y_val) <= 0)/len(y_val))
    plt.figure()
    plt.plot(range(len(w2s)),train,color="#ff7f00")
    plt.plot(range(len(w2s)),val,color="#984ea3")
    plt.show()

    trains = []
    vals = []
    for idx in range(len(all_a)):
        train=[]
        val=[]
        for i in range(len(all_a[idx])):
            train.append(sum(sum(((all_a[idx][i]*Ks[i].T)*y).T) <= 0)/len(y))
            val.append(sum(sum(((all_a[idx][i]*K_vals[i].T)*y_val).T) <= 0)/len(y_val))
        trains.append(train)
        vals.append(val)
    for train, val in zip(trains, vals):
        plt.figure()
        plt.plot(range(len(w1s)),train,color="#ff7f00")
        plt.plot(range(len(w1s)),val,color="#984ea3")
        plt.show()





