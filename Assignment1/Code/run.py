# Code for running all of the different learners

import logging
import numpy as np
import matplotlib.pyplot as plt

from tests.DecisionTreeTest import DecisionTreeTests
from tests.KNNTest import KNNTests
from tests.SVMTest import SVMTests
from tests.NNTest import NNTests

from util import loading_data
from util.graphing import plot_helper
from tests.BaseTest import TestDetails


# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def grouped_bar_chart(labels, data_1, data_2, label_1, label_2, yaxis_label,
                      title='', file_name=''):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    x = np.arange(len(labels)).astype(float)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(4)
    rects1 = ax.bar(x - width/2, data_1, width, label=label_1)
    rects2 = ax.bar(x + width/2, data_2, width, label=label_2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(yaxis_label)
    ax.legend()

    plot_helper(title,file_name, show=False)


def graph_runtimes(train_times, test_times, dataset_name, seed):

    labels = list(train_times.keys())
    train_times_list = list(train_times.values())
    test_times_list = list(test_times.values())
    grouped_bar_chart(labels, train_times_list, test_times_list, 'Test', 'Train', "Runtime (s)", 'Time to train and test each algorithm','runtimes_for_' + dataset_name )
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
    dataset_2 = 0

    test_1_dataset = TestDetails(dataset_1, seed)

    train_times = {}
    test_times = {}

    # run experiments
    run_experiment(DecisionTreeTests, test_1_dataset, train_times, test_times)
    run_experiment(KNNTests, test_1_dataset, train_times, test_times)
    # run_experiment(NNTests, test_1_dataset, train_times, test_times)
    run_experiment(SVMTests, test_1_dataset, train_times, test_times)

    graph_runtimes(train_times, test_times, 'lung_cancer', seed)







