import numpy as np
import csv
from collections import Counter, OrderedDict

def read_file(file):
    """
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
    def __init__(self, file, test=False, norm = None):
        self.test=test
        data = read_file(file)
        del data['id']
        day = []
        month = []
        year = []
        for date in data['date']:
            split = date.split('/')
            try:
                assert len(split)==3
            except:
                raise Exception("Date not valid: " + date)
            day.append(split[1])
            month.append(split[0])
            year.append(split[2])
        del data['date']
        data['day'] = day
        data['month'] = month
        data['year'] = year
        for k, v in data.items():
            data[k] = [float(x) for x in v]
        self.data = data.copy()
        if norm is None:
            self.norm = OrderedDict()
        else:
            self.norm = norm
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
        if not test:
            self.y = np.array(self.data['price'])
            self.y_norm = np.array(self.data_norm['price'])
            del self.data['price']
            del self.data_norm['price']
        self.keys = list(self.data.keys())
        X = np.array(list(self.data.values()))
        X_norm = np.array(list(self.data_norm.values()))
        c = 0
        for k in self.keys:
            try:
                assert list(X[c]) == self.data[k]
                assert list(X_norm[c]) == self.data_norm[k]
            except:
                raise Exception("Error when loading numpy array: " + k + " column not the same in dict and array")
            c+=1
        self.X = np.transpose(X)
        self.X_norm = np.transpose(X_norm)
        self.data
        self.data_norm

    def get_stats(self, norm=False):
        if norm:
            print_stats(self.data_norm)
            if self.test:
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
            if self.test:
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

    def denormalize(self, y, type='price'):
        min_val, max_val = self.norm[type]
        for idx in range(len(y)):
            if max_val-min_val>0:
                y[idx] = y[idx]*(max_val-min_val)+min_val
            else:
                y[idx] = y[idx]*max_val











