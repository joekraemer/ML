import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder


# root_dir = '/root/ML/'
root_dir = '/Users/wchen/PycharmProjects/ML/'


class Dataset(object):
    def __init__(self, train_x, train_y, test_x, test_y, name):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.name = name
        pass


def load_lung_cancer(split=0.75, balance=False):
    data_df = pd.read_csv(root_dir + 'Assignment1/Code/Datasets/survey lung cancer.csv')
    data = data_df.to_numpy()

    # convert YES/NO strings to 1 or 0
    data[:, -1:] = np.where(data[:, -1:] == 'NO', 0, 1)

    # convert M/F strings to 1 or 0
    data[:, 0] = np.where(data[:, 0] == 'M', 0, 1)

    X = data[:, 0:-1]
    y = data[:, -1].astype('int')

    if balance:
        # TODO: This data set is unbalanced
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        X_balanced = X_resampled
        y_balanced = y_resampled

    else:
        X_balanced = X
        y_balanced = y

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

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    ds = Dataset(X_train_scaled, y_train_hot, X_test_scaled, y_test_hot, name)
    return ds


def load_red_wine():
    red_wine_path = root_dir + 'Assignment1/Code/Datasets/Wine/winequality-red.csv'
    return load_wine(red_wine_path, 'red_wine')


def load_white_wine():
    red_wine_path = root_dir + 'Assignment1/Code/Datasets/Wine/winequality-white.csv'
    return load_wine(red_wine_path, 'White Wine')


def load_yeast():
    data_pd = pd.read_table('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Yeast/yeast.data')
    data_pd2 = pd.read_table('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Yeast/yeast.data', sep="\t")
    return data_pd


def load_cardio():
    ''' This is a continousu dataset '''
    dataset_path = '/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/Cardiotocography/TripleCatCSV.csv'
    data_df = pd.read_csv(dataset_path)

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

    ds = Dataset(X_train_scaled, y_train, X_test_scaled, y_test, 'Cardio')
    df = pd.DataFrame(X).astype(float)
    # plot_correlation_matrix(df, ds.name, hue=10)
    return ds

def load_diabetic():
    data = arff.loadarff(root_dir + 'Assignment1/Code/Datasets/DiabeticRetinopathy/messidor_features.arff')[0]
    data_df = pd.DataFrame(data).astype(float)

    data = data_df.to_numpy()

    X = data[:, 0:-1]
    y = data[:, -1].astype(int)

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    X_balanced = X_resampled
    y_balanced = y_resampled

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state=42)

    # Standardize the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Scale only on the training data so we don't leak into the test
    X_test_scaled = scaler.fit_transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


    ds = Dataset(X_train_scaled, y_train_hot, X_test_scaled, y_test_hot, 'diabetic')
    return ds


def load_student_entrance_exam():
    data = arff.loadarff('/Users/wchen/PycharmProjects/ML/Assignment1/Code/Datasets/StudentEntranceExam/CEE_DATA.arff')[0]
    data_df = pd.DataFrame(data)

    data = data_df.to_numpy()

    X = data[:, 1:]
    y = data[:, 0]

    # TODO need to remove the b' portion of the string
    # TODO need to make sure that the Excellent, vg, etc are configured correctly to be ordinal

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


if __name__ == "__main__":
    # testing one-hot encoding
    load_red_wine()



