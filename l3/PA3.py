import pandas as pd
import zipfile as zf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import time
from collections import Counter


def loadzip(zipname, csvname):
    """
    This function imports a csv from a zipfile and extracts, processes and returns a dataframe from it.

    :param zipname: A string containing the path/filename for the zip you want to extract from
    :param csvname: A string containing the filename for the csv file inside the above zip
    """
    reader = zf.ZipFile(zipname)
    df = pd.read_csv(reader.open(csvname))
    return df

class make_node():
    def __init__(self, index, feature_dict):
        self.left = None
        self.right = None
        self.index = index
        self.feature_dict = feature_dict
        self.value = None
        self.feature = None


class make_tree():
    def __init__(self, df, max_depth, n_features = 0, alphas = None):
        self.y = df['class']
        self.df = df
        self.n_features = n_features
        self.max_depth = max_depth
        root_dict = {}
        for key in df:
            if len(df[key].unique())>1 and key != "class":
                root_dict[key] = list(df[key].unique())
        self.root = make_node(list(df.index),root_dict)
        root_depth = 0
        self.queue = [(self.root, root_depth, alphas)]
        while len(self.queue) > 0:
            self.split_node(*self.queue.pop(0))
        del self.df

    def split_node(self, node, depth, alphas = None):
        cur_depth = depth + 1
        df_subset = self.df.iloc[node.index]
        df_len = len(df_subset)
        if df_len == 0:
            print(node.index)
            print(depth)
            return
        if alphas is None:
            cur_gini = float(sum(df_subset['class']))/df_len
        else:
            pos_alpha = sum([alphas[a] for a in df_subset[df_subset['class'] == 1].index])
            all_alpha = sum([alphas[a] for a in node.index])
            cur_gini = float(pos_alpha)/all_alpha
        cur_gini = 1-(cur_gini)**2-(1-cur_gini)**2
        max_gain = 0
        max_feature = None
        max_val = None
        if cur_depth > self.max_depth or cur_gini == 0:
            return
        else:
            if self.n_features > 0:
                if len(node.feature_dict.keys()) < self.n_features:
                    n_feat = len(node.feature_dict.keys())
                else:
                    n_feat = self.n_features
                features = np.random.choice(list(node.feature_dict.keys()), n_feat, replace=False)
                #print(features)
            else:
                features = list(node.feature_dict.keys())
            for feature in features:
                if len(node.feature_dict[feature])==2:
                    value = node.feature_dict[feature][0]
                    l_subset = df_subset['class'][df_subset[feature] == value]
                    r_subset = df_subset['class'][df_subset[feature] != value]
                    if len(l_subset) != 0 and len(r_subset) != 0:
                        if alphas is None:
                            gini_l = float(sum(l_subset))/len(l_subset)
                            gini_r = float(sum(r_subset))/len(r_subset)
                        else:
                            pos_l = sum([alphas[a] for a in l_subset[l_subset == 1].index])
                            all_l = sum([alphas[a] for a in l_subset.index])
                            gini_l = float(pos_l)/all_l
                            pos_r = sum([alphas[a] for a in r_subset[r_subset == 1].index])
                            all_r = sum([alphas[a] for a in r_subset.index])
                            gini_r = float(pos_r)/all_r 
                        gini_l = 1-gini_l**2-(1-gini_l)**2
                        gini_r = 1-gini_r**2-(1-gini_r)**2
                        gain = cur_gini - (float(len(l_subset))/df_len)*gini_l - (float(len(r_subset))/df_len)*gini_r
                        if gain > max_gain:
                            max_gain = gain
                            max_feature = feature
                            max_val = value
                else:
                    for value in node.feature_dict[feature]:
                        l_subset = df_subset['class'][df_subset[feature] == value]
                        r_subset = df_subset['class'][df_subset[feature] != value]
                        if len(l_subset) != 0 and len(r_subset) != 0:
                            if alphas is None:
                                gini_l = float(sum(l_subset))/len(l_subset)
                                gini_r = float(sum(r_subset))/len(r_subset)
                            else:
                                pos_l = sum([alphas[a] for a in l_subset[l_subset == 1].index])
                                all_l = sum([alphas[a] for a in l_subset.index])
                                gini_l = float(pos_l)/all_l
                                pos_r = sum([alphas[a] for a in r_subset[r_subset == 1].index])
                                all_r = sum([alphas[a] for a in r_subset.index])
                                gini_r = float(pos_r)/all_r
                            gini_l = 1-gini_l**2-(1-gini_l)**2
                            gini_r = 1-gini_r**2-(1-gini_r)**2
                            gain = cur_gini - (float(len(l_subset))/df_len)*gini_l - (float(len(r_subset))/df_len)*gini_r
                            if gain > max_gain:
                                max_gain = gain
                                max_feature = feature
                                max_val = value
            if max_feature is not None:
                #print("max_gain:"+str(max_gain))
                node.feature = max_feature
                node.value = max_val
                new_dict = node.feature_dict.copy()
                new_dict[max_feature].remove(max_val)
                if len(new_dict[max_feature]) < 2:
                    del new_dict[max_feature]
                l_index = list(df_subset[df_subset[max_feature] == max_val].index)
                r_index = list(df_subset[df_subset[max_feature] != max_val].index)
                #print("l_index:"+str(len(l_index)))
                #print("r_index:"+str(len(r_index)))
                node.left = make_node(l_index,new_dict)
                node.right = make_node(r_index,new_dict)
                self.queue.append((node.left,cur_depth,alphas))
                self.queue.append((node.right,cur_depth,alphas))

    def predict(self, df):
        preds = [None]*len(df)
        for idx in range(len(df)):
            node = self.root
            while node.feature is not None:
                if df[node.feature][idx] == node.value:
                    node = node.left
                else:
                    node = node.right
            preds[idx] = self.y.iloc[node.index].mode()[0]
        return preds

    def print_nodes(self, node, p_string):
        if node.feature is None:
            print(p_string)
            print(self.y.iloc[node.index].value_counts())
        else:
            self.print_nodes(node.left, p_string + "(" + node.feature + "=" + str(node.value) + ")->")
            self.print_nodes(node.right, p_string + "(" + node.feature + "!=" + str(node.value) + ")->")

    def print_tree(self):
        self.print_nodes(self.root, '')
        





class make_forest():
    def __init__(self, df, max_depth, features, trees):
        self.trees = [None]*trees
        self.n = trees
        for idx in range(trees):
            boot_df = df.sample(len(df), replace=True).reset_index(drop=True)
            self.trees[idx] = make_tree(boot_df, max_depth, features)

    def predict(self,df):
        preds = [None]*self.n
        n = float(self.n)
        for idx in range(self.n):
            preds[idx] = self.trees[idx].predict(df)
        voted_preds = [Counter(x).most_common()[0][0] for x in zip(*preds)]
        return voted_preds

class ada_boost():
    def __init__(self, df, n_models, max_depth):
        self.df = df
        self.n = n_models
        self.models = [None]*n_models
        self.weights = [None]*n_models
        alphas = [1]*len(df)
        for idx in range(n_models):
            self.models[idx] = make_tree(df, max_depth, 0, alphas)
            pred = self.models[idx].predict(df)
            pos_alpha = float(sum([alphas[a] for a in df[df['class'] == pred].index]))
            all_alpha = sum(alphas)
            weighted_err = 1-(pos_alpha/all_alpha)
            update_alpha = (1./2)*math.log((1-weighted_err)/weighted_err)
            for i in range(len(df)):
                if pred[i] == df['class'][i]:
                    alphas[i] = alphas[i]*(math.e**-update_alpha)
                else:
                    alphas[i] = alphas[i]*(math.e**update_alpha)
            self.weights[idx] = update_alpha

    def predict(self, df):
        preds = [None]*self.n
        total_weights = sum(self.weights)
        for idx in range(self.n):
            preds[idx] = self.models[idx].predict(df)
        voted_preds = [math.floor(float(np.dot(x,self.weights))/total_weights+.5) for x in zip(*preds)]
        return voted_preds




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
    
    if 1 in args.parts:
        trees = [None]*8
        pred = [None]*8
        pred_val = [None]*8
        train_err = [None]*8
        val_err = [None]*8
        for idx in range(8):
            trees[idx] = make_tree(df,idx + 1)
            pred[idx] = trees[idx].predict(df)
            pred_val[idx] = trees[idx].predict(df_val)
            train_err[idx] = 1-float(sum(df['class'] == pred[idx]))/len(pred[idx])
            val_err[idx] = 1-float(sum(df_val['class'] == pred_val[idx]))/len(pred_val[idx])
            if idx == 1:
                print("The training error for a tree with " + str(idx+1) + " depth is: " + str(train_err[idx]))
                print("The validation error for a tree with " + str(idx+1) + " depth is: " + str(val_err[idx]))
        if not args.hide:
            plt.figure()
            plt.plot(range(8),train_err,color="#ff7f00", label="Train")
            plt.plot(range(8),val_err,color="#984ea3", label="Validation")
            plt.title("Decision Trees")
            plt.xlabel("Depth")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    if 2 in args.parts:
        pass

    if 3 in args.parts:
        pass

