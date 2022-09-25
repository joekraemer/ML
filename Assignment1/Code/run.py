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


class RedWineTest(TestDetails):
    def __init__(self,  ds, seed):
        super().__init__(ds, seed)

        # Boosted Tests Consts
        self.boost_n_estimators_list = np.linspace(1, 80, 40).astype(int)
        self.boost_lr_list = np.linspace(0.01, 1.25, 20)
        # Boosted Learner Consts
        self.boost_n_estimators = 65
        self.boost_learning_rate = 1

        # DT Tests Consts
        self.DTT_leaf_sizes_list = np.linspace(1, 100, 100, dtype=int)
        self.DTT_ccp_alpha_list = np.linspace(0.0, 0.005, 30)
        self.DTT_max_depth_list = np.linspace(1, 20, 20, dtype=int)

        # DT Learner Consts
        self.DTL_leaf_size = 1
        self.DTL_ccp_alpha = 0.0
        self.DTL_max_depth = 6

        # KNN Consts

        # KNN Learner Consts
        self.KNNL_n_neighbors = 15
        self.KNNL_algorithm = 'ball_tree'

        # NN Test Consts
        # NN Learner Consts
        self.NNL_alpha = 1e-5
        self.NNL_n_nodes = 50
        self.NNL_n_layers = 3

        # SVM Tests Consts
        # SVM Learner Consts
        self.SVML_C = 1
        self.SVML_kernel = 'rbf'


class CardioTest(TestDetails):
    def __init__(self,  ds, seed):
        super().__init__(ds, seed)

        # Boosted Tests Consts
        self.boost_n_estimators_list = np.linspace(1, 80, 10).astype(int)
        self.boost_lr_list = np.linspace(0.01, 1.25, 10)
        # Boosted Learner Consts
        self.boost_n_estimators = 65
        self.boost_learning_rate = 1

        # DT Tests Consts
        self.DTT_leaf_sizes_list = np.linspace(1, 100, 10, dtype=int)
        self.DTT_ccp_alpha_list = np.linspace(0.0, 0.005, 10)
        self.DTT_max_depth_list = np.linspace(1, 20, 10, dtype=int)

        # DT Learner Consts
        self.DTL_leaf_size = 1
        self.DTL_ccp_alpha = 0.0
        self.DTL_max_depth = 6

        # KNN Consts

        # KNN Learner Consts
        self.KNNL_n_neighbors = 15
        self.KNNL_algorithm = 'ball_tree'

        # NN Test Consts
        # NN Learner Consts
        self.NNL_alpha = 1e-5
        self.NNL_n_nodes = 50
        self.NNL_n_layers = 3

        # SVM Tests Consts
        # SVM Learner Consts
        self.SVML_C=1
        self.SVML_kernel='rbf'



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

    grouped_bar_chart(labels, train_times_mean, test_times_mean, 'Train', 'Test', "Runtime (ms)", d1_std=train_times_std, d2_std=test_times_std, title='Time to train and test each algorithm',file_name='runtimes_for_' + dataset_name, folder=dataset_name )
    return


def run_all_exp_for_dataset(test_dataset):
    train_times = {}
    test_times = {}

    # run experiments
    run_experiment(BoostedTests, test_dataset, train_times, test_times)
    run_experiment(DecisionTreeTests, test_dataset, train_times, test_times)
    run_experiment(KNNTests, test_dataset, train_times, test_times)
    run_experiment(NNTests, test_dataset, train_times, test_times)
    run_experiment(SVMTests, test_dataset, train_times, test_times)

    graph_runtimes(train_times, test_times, test_dataset.ds.name, seed)
    return


def run_experiment(test, test_dataset, train_times, test_times):
    test_instance = test(test_dataset)
    dt_runtimes = test_instance.run()
    train_times[test_instance.Name] = dt_runtimes[0]
    test_times[test_instance.Name] = dt_runtimes[1]
    return


if __name__ == '__main__':
    seed = 123456
    print("Seed {}".format(seed))

    print("Load datasets\r\n")

    # dataset_1 = loading_data.load_lung_cancer()
    # dataset_2 = loading_data.load_red_wine()
    # dataset_3 = loading_data.load_student_entrance_exam()
    # dataset_4 = loading_data.load_absenteeism_at_work()
    # dataset_5 = loading_data.load_yeast()
    dataset_6 = loading_data.load_cardio()

    print("Starting Tests....")

    #run_all_exp_for_dataset(TestDetails(dataset_1, seed))
    #run_all_exp_for_dataset(TestDetails(dataset_2, seed))
    run_all_exp_for_dataset(CardioTest(dataset_6, seed))







