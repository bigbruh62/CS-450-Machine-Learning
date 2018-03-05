import pandas as pd
import numpy as np
import random

def main():
    numCategories = 0
    numNeurons = 0
    data = []
    output = []

    print("Please select which data to model:")
    print("\tFor Iris enter 1")
    print("\tFor Pima enter 2")
    database = input("> ")

    if database == 1:
        data, numCategories = iris_data_prep()
        numNeurons = 2
    else:
        data, numCategories = pima_data_prep()
        numNeurons = 1

    brain = neuralNet(numNeurons, numCategories)
    output = brain.get_sums(data[0])

    print(output)

class neuron:
    weights = []
    sum = 0
    threshold = 0

    def __init__(self, numCategories, threshold=0):
        self.threshold = threshold

        i = 0
        while (i < numCategories + 1):
            self.weights.append(0)
            i += 1

    def initialize_weights(self):
        for weight in self.weights:
            weight = random.uniform(-1.0, 1.0)

    def calc_sum(self, inputs):
        i = 0
        for weight in self.weights:
            self.sum += self.weights[i] * inputs[i]

class neuralNet:
    neurons = []
    bias = -1
    threshold = 0

    def __init__(self, numNeurons, numCategories, bias=-1, threshold=0):
        # account for user input of bias
        if bias != -1:
            self.bias = bias

        # account for user input of threshold
        if threshold != 0:
            self.threshold = threshold

        # create the neuron array
        i = 0
        while (i < numNeurons):
            self.neurons.append(neuron(numCategories + 1))
            i += 1

    def get_sums(self, inputs):
        # put the bias value at the head of the list
        inputs = np.insert(inputs, 0, self.bias)

        outputs = []

        i = 0
        for neuron in self.neurons:
            tmp = neuron.calc_sum(inputs)
            if tmp < self.threshold:
                tmp = 0
            else:
                tmp = 1

            outputs.append(tmp)

            i += 1
        return outputs

def iris_data_prep():
    iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

    columns = "sepal_length sepal_width petal_length petal_width class".split()
    iris_data.columns = columns

    iris_data_targets = iris_data.iloc[:, 4:]
    iris_data = iris_data.iloc[:, :4]

    numCol = len(iris_data.columns)
    iris_data_array = iris_data.as_matrix()

#    print(iris_data_array)
    return iris_data_array, numCol

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

    numCol = len(pima_data.columns)

#    print(pima_data_array)
    return pima_data_array, numCol

main()