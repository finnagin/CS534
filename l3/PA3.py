import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import time


def loadzip(zipname, csvname):
    """
    This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

    :param zipname: A string containing the path/filename for the zip you want to extract from
    :param csvname: A string containing the filename for the csv file inside the above zip
    """
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname))
    return df

class node():
    def __init__(self, indexes, feature_dict):
        self.left = None
        self.right = None
        self.indexes = indexes
        self.feature_dict = feature_dict


class tree():
    def __init__(self, df, max_depth, n_features = 0):
        self.df = df
        self.n_features = n_features
        self.max_depth = max_depth
        root_dict = {}
        for key in df:
            if len(df[key].unique())>1:
                root_dict[key] = df[key].unique()
        self.root = node(list(df.indexes))
        root_depth = 0
        self.queue = [(self.root, cur_depth)]
        while len(self.queue) > 0:
            self.split_node(*self.queue.pop(0))

    def split_node(self, node, depth):
        cur_depth = depth + 1
        if cur_depth > self.max_depth:
            return
        else:
            if self.n_features > 0:
                features = np.choice(list(node.feature_dict.keys()), self.n_features, replace=False)
            else:
                features = list(node.feature_dict.keys())
            for feature in features:


class forrest():
    def __init__(self, df, max_depth, features, trees):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
    parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

    args = parser.parse_args()

    seed = int(time.time())
    np.random.seed(seed-1)

    # Load the data from the zips
    df = loadzip('data/pa3_train.zip','pa3_train.csv')
    df_val = loadzip('data/pa3_val.zip','pa3_val.csv')
    df_test = loadzip('data/pa3_test.zip','pa3_test.csv')