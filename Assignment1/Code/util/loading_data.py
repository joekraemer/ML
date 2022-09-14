import pandas as pd
import numpy as np
from scipy.io import arff
from tests.BaseTest import TestDetails
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class Dataset(object):
    def __init__(self, train_x, train_y, test_x, test_y, name):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.name = name
        pass


def load_lung_cancer(split=0.75):
    data_df = pd.read_csv('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/survey lung cancer.csv')
    data = data_df.to_numpy()

    # convert YES/NO strings to 1 or 0
    data[:, -1:] = np.where(data[:, -1:] == 'NO', 0, 1)

    # convert M/F strings to 1 or 0
    data[:, 0] = np.where(data[:, 0] == 'M', 0, 1)

    X = data[:, 0:-1]
    y = data[:, -1].astype('int')

    # TODO: This data set is unbalanced
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_balanced = X_resampled
    y_balanced = y_resampled

    # TODO: Not sure if this is how I should split the data, people were talking about balancing or something
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state=42)

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale only on the training data so we don't leak into the test
    X_test_scaled = scaler.fit_transform(X_test)

    ds = Dataset(X_train_scaled, y_train, X_test_scaled, y_test, 'Lung Cancer')
    return ds


def load_wine(path, name):
    ''' This is a continousu dataset '''
    data_df = pd.read_csv(path, sep=';')

    data = data_df.to_numpy()

    X = data[:, 0:-1]
    y = data[:, -1]

    # TODO: This data set is unbalanced
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_balanced = X_resampled
    y_balanced = y_resampled

    # TODO: Not sure if this is how I should split the data, people were talking about balancing or something
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state=42)

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale only on the training data so we don't leak into the test
    X_test_scaled = scaler.fit_transform(X_test)

    ds = Dataset(X_train_scaled, y_train, X_test_scaled, y_test, name)
    return ds


def load_red_wine():
    red_wine_path = '/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Wine/winequality-red.csv'
    return load_wine(red_wine_path, 'Red Wine')


def load_white_wine():
    red_wine_path = '/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Wine/winequality-white.csv'
    return load_wine(red_wine_path, 'White Wine')


def load_yeast():
    data_pd = pd.read_table('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Yeast/yeast.data')


def load_student_entrance_exam():
    data = arff.loadarff('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/StudentEntranceExam/CEE_DATA.arff')[0]
    data_df = pd.DataFrame(data)

    data = data_df.to_numpy()

    X = data[:, 1:]
    y = data[:, 0]

    # TODO need to remove the b' portion of the string
    # TODO need to make sure that the Exellent, vg, etc are configured correctly to be ordinal

    ordinal_enc = preprocessing.OrdinalEncoder()
    post_ord = ordinal_enc.fit_transform(X, y)
    # TODO: Convert all this categorical data to things that sklearn can use
    print('stop')

    return


def load_absenteeism_at_work():
    path = '/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Absenteeism_at_work_AAA/Absenteeism_at_work.csv'
    data_df = pd.read_csv(path, sep=';')

    data = data_df.to_numpy()

    X = data[:, 0:-1]
    y = data[:, -1]


    # TODO: This data set is unbalanced
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_balanced = X_resampled
    y_balanced = y_resampled

    # TODO: Not sure if this is how I should split the data, people were talking about balancing or something
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state=42)

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale only on the training data so we don't leak into the test
    X_test_scaled = scaler.fit_transform(X_test)

    ds = Dataset(X_train_scaled, y_train, X_test_scaled, y_test, 'absentee')
    return ds




def loading_arff():
    '''Example from documentation'''
    # data = arff.load(open('../data/data.arff', 'r'))['data']
    X = [i[:4] for i in data]
    y = [i[-1] for i in data]



