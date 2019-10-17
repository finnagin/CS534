import numpy as np
import csv
from collections import Counter

def read_file(file):
    with open(file) as fid:
            data={}
            csv_reader = csv.reader(fid)
            first = True
            for row in csv_reader:
                if first:
                    feature_keys = row
                    first = False
                    for key in feature_keys:
                        data[key] = []
                else:
                    for idx in range(len(feature_keys)):
                        data[feature_keys[idx]].append(row[idx])
    return data

def print_stats(data):
    for k, v in data.items():
        print(k, ":")
        if k in ['condition', 'grade', 'waterfront']:
            counts = Counter(v).most_common()
            for ck, cv in counts:
                if str(ck).endswith(".0"):
                    print("  ", str(ck).replace(".0",""), ": ", cv)
                else:
                    print("  ", ck, ": ", cv)
        else:
            min_val = str(min(v))
            max_val = str(max(v))
            if min_val.endswith(".0"):
                min_val = min_val.replace(".0","")
            if max_val.endswith(".0"):
                max_val = max_val.replace(".0","")
            print("  Mean: ", np.mean(v))
            print("  Standard Deviation: ", np.std(v))
            print("  Range: [", min_val, ",", max_val, "]")





class preprocess():
    def __init__(self, file, test=False):
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
                print("Date not valid: ", date)
                day.append('0')
                month.append('0')
                year.append('0')
            day.append(split[1])
            month.append(split[0])
            year.append(split[2])
        del data['date']
        data['day'] = day
        data['month'] = month
        data['year'] = year
        for k, v in data.items():
            data[k] = [float(x) for x in v]
        print_stats(data)
        self.data = data.copy()
        for k, v in data.items():
            max_val = max(v)
            data[k] = [x/max_val for x in v]
        self.data_norm = data.copy()
        if test:
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
                print("  Mean: ", np.mean(self.y_norm))
                print("  Standard Deviation: ", np.std(self.y_norm))
                print("  Range: [", min_val, ",", max_val, "]")
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
                print("  Mean: ", np.mean(self.y))
                print("  Standard Deviation: ", np.std(self.y))
                print("  Range: [", min_val, ",", max_val, "]")










