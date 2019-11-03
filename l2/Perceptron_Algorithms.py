import pandas as pd
import zipfile as zf
import numpy as np

def loadzip(zipname, csvname, test=False):
    """
    This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

    :param zipname: A string containing the path/filename for the zip you want to extract from
    :param csvname: A string containing the filename for the csv file inside the above zip
    :param test: A boolean indicating if the zip contains the test data
    """
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname), header=None)
    # If not ttest then convert the y values
    if not test:
        df[0] = np.where(df[0] == 3, 1,df[0])
        df[0] = np.where(df[0] == 5, -1,df[0])
    # Add the bias term
    df[df.shape[1]] = [1.]*len(df)
    return df

def online_perceptron_loop(y, X, iters):
    """
    This function trains a online perceptron model X and y 

    :param y: A 1D numpy array containing the y values for training
    :param X: A 2D numpy array containing the feature vectors

    :return: A list of numpy arrays for each iterations w vector
    """
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

def avg_perceptron_loop(y, X, iters):
    """
    This function trains a average perceptron model on X and y with power p ending in iters iterations. 

    :param y: A 1D numpy array containing the y values for training
    :param X: A 2D numpy array containing the feature vectors

    :return: A list of numpy arrays for each iterations w vector
    """
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

# Depriciated
#def kernel_perceptron_loop(df, iters, p):
#    X = np.array(df.values[:,1:])
#    y = np.array(df[0].values)
#    n = X.shape[0]
#    a = np.zeros(n)
#    K = (1+np.matmul(X,X.T))**p
#    a_list = []
#    for i in range(iters):
#        a_list.append(a.copy())
#        for idx in range(n):
#            u = sum(a*K[idx]*y)
#            if u*y[idx] <= 0:
#                a[idx] += 1
#    a_list.append(a.copy())
#    return a_list

def KernelizedPerceptron(y, X, p, iters):
    """
    This function trains a polynomial kernel perceptron model on X and y with power p ending in iters iterations. 

    :param y: A 1D numpy array containing the y values for training
    :param X: A 2D numpy array containing the feature vectors
    :param p: A positive integer containing the power you want to use for the kernel

    :return: A list of numpy arrays for each iterations alpha values
    """
    (num_of_examples, feature_number) = X.shape
    # Initialize alpha and list of alphas
    a = np.zeros(num_of_examples)
    weight_list = []
    weight_list.append(np.copy(a))
    K = np.power(1 + np.matmul(X, np.transpose(X)),p)
    iter = 0
    while iter < iters:
        for t in range(num_of_examples):
            u = np.dot(a, np.transpose(K[:,t])*y)
            # If mistake is made add 1 to alpha_t
            if y[t]*u <= 0:
                a[t] += 1
        # append alpha vector for this iteration
        weight_list.append(np.copy(a))
        iter += 1
    return weight_list

def predictK(a, K, y, X_test):
    """
    Predicts the values for X_test using the corresponding matrix K, the training y vector, and the alpha vector

    :param a: A 1D numpy array containing the apha values
    :param K: A 2D  numpy array containing the Gram matrix generated from the testig feature matrix and the training feature matrix
    :param y: A 1D numpy array containing the y values used to train the alpha vector
    :param X_test: A 2D numpy array containing the feature vectors you want to predict on
    """
    preds = np.sign(np.dot(K,a*y))
    return preds