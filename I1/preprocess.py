import numpy as np
import csv
from collections import Counter, OrderedDict

def read_file(file):
    """
    This fuction reads the csv into an ordered dict

    :param file: a string containing the file path to the csv you wish to load
    :return: a dict containg the columns of the csv with the headers as the key
    """
    with open(file) as fid:
            data=OrderedDict()
            csv_reader = csv.reader(fid)
            first = True
            for row in csv_reader:
                # check if the first row (header row)
                if first:
                    feature_keys = row
                    first = False
                    # initialize column list
                    for key in feature_keys:
                        data[key] = []
                else:
                    # append each element to the corresponding column key
                    for idx in range(len(feature_keys)):
                        data[feature_keys[idx]].append(row[idx])
    return data

def print_stats(data):
    """
    This fuction prints the stats for the values in the data dict

    :param data: a dict containing the data loaded from the csv that has been processed to contain float values
    :return: Stastics about each feature printed to the terminal
    """
    for k, v in data.items():
        print(str(k)+":")
        # check if categorical
        if k in ['condition', 'grade', 'waterfront']:
            counts = Counter(v).most_common()
            for ck, cv in counts:
                if str(ck).endswith(".0"):
                    # since these values are floats this removes the trailing .0 for integer values
                    print("  " + str(ck).replace(".0","") + ": " + str(cv))
                else:
                    print("  " + str(ck) + ": "+ str(cv))
        # print stats for non-categorical values
        else:
            min_val = str(min(v))
            max_val = str(max(v))
            if min_val.endswith(".0"):
                min_val = min_val.replace(".0","")
            if max_val.endswith(".0"):
                max_val = max_val.replace(".0","")
            print("  Mean: " + str(np.mean(v)))
            print("  Standard Deviation: " + str(np.std(v)))
            print("  Range: [" + min_val + "," + max_val + "]")





class preprocess():
    """
    This class imports and preprocesses the csv data for the 1st implementation assignment

    :param file: A string containing a path to the csv containing the data
    :param test: A boolean if True then it will assume there is no price feature, if False then it will assume there is. (defaults to False)
    :param norm: A dict containing the max and min values for each feature. If set to None it will infer this from the csv.
    :attr X: A numpy array containing all of the data in the csv besides price
    :attr X_norm: A numpy array containing a normalized version of X
    :attr y: A numpy array containing all the price values from the csv
    :attr y_norm: A numpy array containing a normalized version of y
    :attr keys: A list of headers for the columns in X and X_norm
    :attr norm: A dict containing the max and min values for each feature.
    """
    def __init__(self, file, test=False, norm = None):
        self.test=test
        data = read_file(file)
        # removes the id feature
        del data['id']
        day = []
        month = []
        year = []
        # splits date into day, month, and year
        for date in data['date']:
            split = date.split('/')
            try:
                assert len(split)==3
            except:
                raise Exception("Date not valid: " + date)
            day.append(split[1])
            month.append(split[0])
            year.append(split[2])
        # removes the date feature and adds day, month, and year
        del data['date']
        data['day'] = day
        data['month'] = month
        data['year'] = year
        # Convert to float values
        for k, v in data.items():
            data[k] = [float(x) for x in v]
        self.data = data.copy()
        # save the values needed for normalization (or prepare to)
        if norm is None:
            self.norm = OrderedDict()
        else:
            self.norm = norm
        # normalize the data
        for k, v in data.items():
            if norm is None:
                max_val = max(v)
                min_val = min(v)
                self.norm[k] = (min_val, max_val)
            else:
                min_val, max_val = norm[k]
            if max_val-min_val>0:
                data[k] = [(x-min_val)/(max_val-min_val) for x in v]
            else:
                data[k] = [x/max_val for x in v]
        self.data_norm = data.copy()
        # cut the price values from X and put then into a y variable
        if not test:
            self.y = np.array(self.data['price'])
            self.y_norm = np.array(self.data_norm['price'])
            del self.data['price']
            del self.data_norm['price']
        self.keys = list(self.data.keys())
        self.y_key = 'price'
        X = np.array(list(self.data.values()))
        X_norm = np.array(list(self.data_norm.values()))
        c = 0
        # verify that data was converted correctly
        for k in self.keys:
            try:
                assert list(X[c]) == self.data[k]
                assert list(X_norm[c]) == self.data_norm[k]
            except:
                raise Exception("Error when loading numpy array: " + k + " column not the same in dict and array")
            c+=1
        # save X, X_norm, y, and y_norm as attributes
        self.X = np.transpose(X)
        self.X_norm = np.transpose(X_norm)
        self.data
        self.data_norm

    def get_stats(self, norm=False):
        """
        A method that prints the stats for the data in this class

        :param norm: A boolean when input as True will print the stats for the normalized data and when false will print the stats for the unnormalized data
        """
        if norm:
            print_stats(self.data_norm)
            if not self.test:
                print("Price:")
                min_val = str(min(self.y_norm))
                max_val = str(max(self.y_norm))
                if min_val.endswith(".0"):
                    min_val = min_val.replace(".0","")
                if max_val.endswith(".0"):
                    max_val = max_val.replace(".0","")
                print("  Mean: "+ str(np.mean(self.y_norm)))
                print("  Standard Deviation: " + str(np.std(self.y_norm)))
                print("  Range: [" + min_val + "," + max_val + "]")
        else:
            print_stats(self.data)
            if not self.test:
                print("Price:")
                min_val = str(min(self.y))
                max_val = str(max(self.y))
                if min_val.endswith(".0"):
                    min_val = min_val.replace(".0","")
                if max_val.endswith(".0"):
                    max_val = max_val.replace(".0","")
                print("  Mean: " + str(np.mean(self.y)))
                print("  Standard Deviation: " + str(np.std(self.y)))
                print("  Range: [" + min_val + "," + max_val + "]")

    def denormalize(self, y, key=None):
        """
        This method converts a 1-D normalized array (numpy or list) into a denormalized array

        :param y: A 1-D array of floats
        :param key: the key for the type of data you are converting (defaults to price)
        """
        # set to price as default
        if key is None:
            key = self.y_key
        # get max and min values
        min_val, max_val = self.norm[key]
        y_denorm = y.copy()
        # convert the values back
        for idx in range(len(y)):
            if max_val-min_val>0:
                y_denorm[idx] = y[idx]*(max_val-min_val)+min_val
            else:
                y_denorm[idx] = y[idx]*max_val
        return y_denorm











