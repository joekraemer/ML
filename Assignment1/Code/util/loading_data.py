import pandas as pd
import numpy as np
from tests.BaseTest import TestDetails


class Dataset(object):
    def __init__(self, x, y, train_x, train_y, test_x, test_y, name):
        self.X = x
        self.Y = y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.name = name
        pass


def load_csv_to_df(path):
    """
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    return df


def load_lung_cancer(split=0.6):
    data_df = load_csv_to_df('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/survey lung cancer.csv')
    data = data_df.to_numpy()

    # convert YES/NO strings to 1 or 0
    data[:, -1:] = np.where(data[:, -1:] == 'NO', 0, 1)

    # convert M/F strings to 1 or 0
    data[:, 0] = np.where(data[:, 0] == 'M', 0, 1)

    #TODO How to split this data up how they want
    # we can create a random order of the data and then split the order in the same place
    random_selection = np.random.choice(len(data), size=len(data), replace=False)

    # compute how much of the data is training and testing
    num_train_rows = int(split * data.shape[0])
    num_test_rows = data.shape[0] - num_train_rows

    np.random.shuffle(data)
    # separate out training and testing data
    train_x = data[:num_train_rows, 0:-1]
    train_y = data[:num_train_rows, -1].astype('int')
    test_x = data[num_test_rows:, 0:-1]
    test_y = data[num_test_rows:, -1].astype('int')

    ds = Dataset(data[:, 0:-1], data[:, -1].astype('int'), train_x, train_y, test_x, test_y, 'Lung Cancer')
    return ds





