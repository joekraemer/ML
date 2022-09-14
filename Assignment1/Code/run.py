# Code for running all of the different learners

import logging
import numpy as np
import matplotlib.pyplot as plt

from tests.DecisionTreeTest import DecisionTreeTests
from tests.KNNTest import KNNTests
from tests.SVMTest import SVMTests
from tests.NNTest import NNTests
from tests.BoostedTest import BoostedTests

from util import loading_data
from util.graphing import plot_helper, grouped_bar_chart
from tests.BaseTest import TestDetails


# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def graph_runtimes(train_times, test_times, dataset_name, seed):

    labels = list(train_times.keys())
    train_times_mean = []
    test_times_mean = []
    train_times_std = []
    test_times_std = []

    for key, run in train_times.items():
        train_times_mean.append(np.mean(run))
        train_times_std.append(np.std(run))

    for key, run in test_times.items():
        test_times_mean.append(np.mean(run))
        test_times_std.append(np.std(run))

    grouped_bar_chart(labels, train_times_mean, test_times_mean, 'Train', 'Test', "Runtime (ms)", d1_std=train_times_std, d2_std=test_times_std, title='Time to train and test each algorithm',file_name='runtimes_for_' + dataset_name )
    return


def run_all_exp_for_dataset(test_dataset):
    train_times = {}
    test_times = {}

    # run experiments
    # run_experiment(DecisionTreeTests, test_dataset, train_times, test_times)
    # run_experiment(KNNTests, test_dataset, train_times, test_times)
    # run_experiment(NNTests, test_dataset, train_times, test_times)
    run_experiment(BoostedTests, test_dataset, train_times, test_times)
    run_experiment(SVMTests, test_dataset, train_times, test_times)

    graph_runtimes(train_times, test_times, 'lung_cancer', seed)
    return


def run_experiment(test, test_dataset, train_times, test_times):
    test_instance = test(test_dataset)
    dt_runtimes = test_instance.run()
    train_times[test_instance.Name] = dt_runtimes[0]
    test_times[test_instance.Name] = dt_runtimes[1]
    return


if __name__ == '__main__':
    seed = np.random.randint(0, 100000, dtype='uint64')
    print("Seed {}".format(seed))

    print("Load datasets\r\n")

    #TODO: Add dataset loading
    dataset_1 = loading_data.load_lung_cancer()
    dataset_2 = loading_data.load_red_wine()
    # dataset_3 = loading_data.load_student_entrance_exam()
    dataset_4 = loading_data.load_absenteeism_at_work()

    test_1_dataset = TestDetails(dataset_1, seed)
    test_2_dataset = TestDetails(dataset_2, seed)

    run_all_exp_for_dataset(test_1_dataset)







