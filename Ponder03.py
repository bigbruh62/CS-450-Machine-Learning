import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


class knnClassifier:

    k = 1
    data = []
    target = []

    def __init__(self, k=1, data=[], target=[] ):
        self.k = k
        self.data = data
        self.target = target

    def fit(self, data, target):
        self.data = data
        self.target = target
        return knnClassifier(self.k, self.data, self.target)

    def predict(self, test_data):
        nInputs = np.shape(test_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            #compute distances
            distances = np.sum((self.data-test_data[n, :])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(self.target[indices[:self.k]])

            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = np.max(counts)


        return closest


    def score(self, x_test, y_test):
        total = len(x_test)
        correct = 0

        for i in range(total):
            if x_test[i] == y_test[i]:
                correct += 1

        return float(correct) / total


def main():
    print("Please enter the number of the data set you would like to analyze\n")
    print("1. UCI Car Evaluation\n")
    print("2. Pima Indian Diabetes\n")
    print("3. Automobile MPG\n")
    dataset = input("> ")

    # preprocess chosen data
    if dataset == 1:
        data, test = car_data_prep()
    elif dataset == 2:
        data, test = pima_data_prep()
    elif dataset == 3:
        data, test = mpg_data_prep()

    # x_train, x_test, y_train, y_test = train_test_split(data, test, test_size=0.3)
    classifier = KNeighborsClassifier(n_neighbors=3)
    kfold = KFold(n_splits=3, random_state=7)
    # kfold_train, kfold_test = kfold.split(data, y=test)
    scores = cross_val_score(classifier, data, test, cv=kfold, scoring='accuracy')

    print(scores)

def car_data_prep():
    uci_car_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")

    # Set column names on data frame
    columns = "buying maint doors persons lug_boot safety target".split()
    uci_car_data.columns = columns

    #print(uci_car_data)
    #print(uci_car_data["persons"].value_counts())

    # make data numerical
    column_numerization = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                           "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                           "doors": {"2": 1, "3": 2, "4": 3, "5more": 4},
                           "persons": {"2": 1, "4": 2, "more": 3},
                           "lug_boot": {"small": 1, "med": 2, "big": 3},
                           "safety": {"low": 1, "med": 2, "high": 3},
                           "target": {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
    uci_car_data.replace(column_numerization, inplace=True)

    # split data and targets into separate data frames
    uci_car_targets = uci_car_data.iloc[:, 6:]
    uci_car_data = uci_car_data.iloc[:, :6]

    # turn the data and target data frames into lists
    uci_car_data_array = uci_car_data.as_matrix()
    uci_car_targets_array = uci_car_targets.as_matrix()
    uci_car_targets_array = uci_car_targets_array.flatten()

    return uci_car_data_array, uci_car_targets_array

def pima_data_prep():
    # get data from the UCI database
    pima_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")

    # give columns names
    columns = "n_preg plasma bp tricep_fold insulin bmi pedigree age target".split()
    pima_data.columns = columns
    #print(pima_data.dtypes)

    # mark zero values as NaN
    pima_data[["n_preg", "plasma", "bp", "tricep_fold", "insulin"]] = pima_data[["n_preg", "plasma", "bp", "tricep_fold", "insulin"]].replace(0, np.NaN)

    # drop rows with NaN
    pima_data.dropna(inplace=True)

    # Split the dataframe into data and targets
    pima_data_targets = pima_data.iloc[:, 8:]
    pima_data = pima_data.iloc[:, :8]

    pima_data_array = pima_data.as_matrix()
    pima_data_targets_array = pima_data_targets.as_matrix()
    pima_data_targets_array = pima_data_targets_array.flatten()

    return pima_data_array, pima_data_targets_array
def mpg_data_prep():
    # get space delimited data set from url
    mpg_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", delim_whitespace=True)
    #print (mpg_data.dtypes)
    # give data set columns names
    columns = "mpg cyl disp hp weight accel year origin model".split()
    mpg_data.columns = columns

    #print(mpg_data.dtypes)

    # replace and remove missing values and associated rows
    mpg_data = mpg_data.replace("?", np.NaN)
    mpg_data.dropna(inplace=True)

    # split dataframe into data (minus the model) and targets
    mpg_targets = mpg_data.iloc[:, :1]
    mpg_data = mpg_data.iloc[:, 1:8]

    mpg_data_array = mpg_data.as_matrix()
    mpg_targets_array = mpg_targets.as_matrix()
    mpg_targets_array = mpg_targets_array.flatten()

    return mpg_data_array, mpg_targets_array

main()