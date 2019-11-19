import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse


def loadzip(zipname, csvname):
    """
    This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

    :param zipname: A string containing the path/filename for the zip you want to extract from
    :param csvname: A string containing the filename for the csv file inside the above zip
    """
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname))
    return df

class tree():
    def __init__(self, df, max_depth, features = 0):
        for d in range(max_depth+1):
            pass

def build_tree(df, max_depth, features = 0):
    pass

class forrest():
    def __init__(self, df, max_depth, features, trees):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", "-p", type=int, nargs='*', help="The parts you want to run this on seperated by spaces", default=[1, 2, 3])
    parser.add_argument("--hide", action='store_true', help="Add if you want to hide the plots")

    args = parser.parse_args()

    # Load the data from the zips
    df = loadzip('data/pa3_train.zip','pa3_train.csv')
    df_val = loadzip('data/pa3_val.zip','pa3_val.csv')
    df_test = loadzip('data/pa3_test.zip','pa3_test.csv')