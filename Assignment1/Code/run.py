# Code for running all of the different learners

import logging

import matplotlib
import numpy as np

from tests.BaseTest import TestDetails
from tests.NNTest import NNTests
from util import loading_data
from util.graphing import grouped_bar_chart, bar_chart

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedWineTest(TestDetails):
    def __init__(self, ds, seed):
        super().__init__(ds, seed)


class DiabeticTest(TestDetails):
    def __init__(self, ds, seed):
        super().__init__(ds, seed)

        # Boosted Tests Consts
        self.boost_n_estimators_list = np.linspace(1, 160, 20).astype(int)
        self.boost_lr_list = np.linspace(0.01, 2, 20)
        self.boost_ccp_alpha_list = np.linspace(0.0, 0.02, 20)

        # Boosted Learner Consts
        self.boost_n_estimators = 45
        self.boost_learning_rate = 1

        # DT Tests Consts
        self.DTT_leaf_sizes_list = np.linspace(1, 150, 20, dtype=int)
        self.DTT_ccp_alpha_list = np.linspace(0.0, 0.003, 5)
        self.DTT_max_depth_list = np.linspace(1, 20, 15, dtype=int)
        # DT Learner Consts
        self.DTL_leaf_size = 1
        self.DTL_ccp_alpha = 0.01
        self.DTL_max_depth = None

        # KNN Consts
        self.KNNT_k_list = np.linspace(1, 80, 20).astype(int)
        # KNN Learner Consts
        self.KNNL_n_neighbors = 30

        # NN Test Consts
        self.NNT_neurons_list = np.linspace(5, 80, 5).astype(int)
        self.NNT_layers_list = np.linspace(1, 6, 5).astype(int)
        self.NNT_alpha_list = np.linspace(1e-6, 1e-4, 4)
        # NN Learner Consts
        self.NNL_n_nodes = 20

        # SVM Tests Consts
        self.SVMT_c_list = np.linspace(0.1, 3, 10)
        # SVM Learner Consts
        self.SVML_C = 1


def graph_runtimes(train_times, test_times, dataset_name, seed):
    labels = list(train_times.keys())

    # prevent from making graphs for a partial run
    if len(labels) < 5:
        return

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

    # convert to seconds
    train_times_mean = train_times_mean * 1000
    train_times_std = train_times_mean * 1000
    test_times_mean = test_times_mean * 1000
    test_times_std = test_times_mean * 1000

    grouped_bar_chart(labels, train_times_mean, test_times_mean, 'Train', 'Test', "Runtime (sec)",
                      d1_std=train_times_std, d2_std=test_times_std, title='', file_name='runtimes_for_' + dataset_name,
                      folder=dataset_name, logy=True)
    return


def graph_final_scores(final_scores, dataset_name):
    labels = list(final_scores.keys())

    # prevent from making graphs for a partial run
    if len(labels) < 5:
        return

    test_scores_mean = []
    test_scores_std = []

    for key, run in final_scores.items():
        test_scores_mean.append(np.mean(run))
        test_scores_std.append(np.std(run))

    bar_chart(labels, test_scores_mean, "F1 Weighted Score", d1_std=test_scores_std, title='',
              file_name='final_scores_' + dataset_name, folder=dataset_name)


def run_all_exp_for_dataset(test_dataset):
    train_times = {}
    test_times = {}
    final_scores = {}

    # run experiments
    # run_experiment(DecisionTreeTests, test_dataset, train_times, test_times, final_scores)
    # run_experiment(BoostedTests, test_dataset, train_times, test_times, final_scores)
    # run_experiment(KNNTests, test_dataset, train_times, test_times, final_scores)
    run_experiment(NNTests, test_dataset, train_times, test_times, final_scores)
    # run_experiment(SVMTests, test_dataset, train_times, test_times, final_scores)

    # graph_runtimes(train_times, test_times, test_dataset.ds.name, seed)
    # graph_final_scores(final_scores, test_dataset.ds.name)
    return


def run_experiment(test, test_dataset, train_times, test_times, final_scores):
    test_instance = test(test_dataset)
    dt_runtimes = test_instance.run()
    train_times[test_instance.Name] = dt_runtimes[0]
    test_times[test_instance.Name] = dt_runtimes[1]
    final_scores[test_instance.Name] = dt_runtimes[2]
    return


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    seed = 123456
    print("Seed {}".format(seed))

    print("Load datasets\r\n")

    # dataset_1 = loading_data.load_red_wine()
    # dataset_2 = loading_data.load_diabetic()
    dataset_3 = loading_data.load_red_wine_unbalanced()

    print("Starting Tests....")

    # run_all_exp_for_dataset(TestDetails(dataset_1, seed))
    # run_all_exp_for_dataset(DiabeticTest(dataset_2, seed))
    run_all_exp_for_dataset(RedWineTest(dataset_3, seed))
