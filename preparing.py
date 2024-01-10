# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import os
import sys
import yaml


def load_data_with_preparing():
    iris = pd.read_csv('iris.csv')
    #print(iris.head())
    #print(np.unique(iris['species']))
    classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    iris['species'] = iris['species'].map(classes)
    #print(iris.head())
    return iris


def main():
    dataset = load_data_with_preparing()
    os.makedirs(os.path.join('data', 'prepared'), exist_ok=True)
    dataset.to_csv('data/prepared/data.csv')

if __name__ == "__main__":
   main()

