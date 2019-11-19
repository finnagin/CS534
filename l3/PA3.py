import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse


def loadzip(zipname, csvname, test=False):
    """
    This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

    :param zipname: A string containing the path/filename for the zip you want to extract from
    :param csvname: A string containing the filename for the csv file inside the above zip
    :param test: A boolean indicating if the zip contains the test data
    """
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname))
    ## If not ttest then convert the y values
    #if not test:
    #    df[0] = np.where(df[0] == 3, 1,df[0])
    #    df[0] = np.where(df[0] == 5, -1,df[0])
    # Add the bias term
    #df[df.shape[1]] = [1.]*len(df)
    return df

class tree():
    def __init__(self, df, depth, features = 0):
        pass

class forrest():
    def __init__(self):
        pass

def build_tree(df)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
    parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

    args = parser.parse_args()

    # Load the data from the zips
    df = loadzip('data/pa3_train.zip','pa3_train.csv')
    df_val = loadzip('data/pa3_val.zip','pa3_val.csv')
    df_test = loadzip('data/pa3_test.zip','pa3_test.csv',True)