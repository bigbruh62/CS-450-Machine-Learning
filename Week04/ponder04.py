import pandas as pd
import numpy as np
from pprint import pprint

# tree classifier class
class decisionTreeClassifier:
    pass


# get house vote data
house_votes = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data")
columns = "Party handi_babies h2o_proj budget phys_fee aid_es relig anti_sat aid_n missile imm synfuels ed sue crime exports s_af_trade".split()
house_votes.columns = columns

house_votes = house_votes.replace("?", np.NaN)
house_votes.dropna(inplace=True)
house_votes_array = house_votes.as_matrix()

#print(house_votes_array)

def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):
    res = entropy(y)

    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    if is_pure(y) or len(y) == 0:
        return y

    gain = np.array([mutual_information(y, x_attr)
                     for x_attr in x.T])
    selected_attr = np.argmax(gain)

    if np.all(gain < 1e-6):
        return y

    sets = partition(x[:, selected_attr])

    res = {}

    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%s = %s" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

y = np.array(['democrat', 'n', 'n', 'y', 'n', 'y', 'n', 'y', 'y', 'n', 'n', 'n', 'n', 'y', 'y', 'y', 'y'])
X = house_votes_array.T
pprint(recursive_split(X, y))