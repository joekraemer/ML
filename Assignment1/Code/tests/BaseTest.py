from abc import ABC, abstractmethod
import datetime
import numpy as np

from sklearn.model_selection import learning_curve, StratifiedShuffleSplit

from util.graphing import plot_learning_curve, plot_scalability
from util.graphing import plot_hyperparam_validation_curve, plot_hyperparam_validation_bar_chart
from sklearn.metrics import f1_score


class TestDetails(object):
    def __init__(self, ds, seed):
        self.ds = ds
        self.seed = seed

        # Boosted Tests Consts
        self.boost_n_estimators_list = np.linspace(1, 80, 40).astype(int)
        self.boost_lr_list = np.linspace(0.01, 1.25, 20)
        self.boost_ccp_alpha_list = np.linspace(0.0, 0.003, 5)

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
        self.KNNT_algo_list = ['ball_tree', 'kd_tree', 'brute']
        self.KNNT_k_list = np.linspace(1, 30, 20).astype(int)

        # KNN Learner Consts
        self.KNNL_n_neighbors = 15
        self.KNNL_algorithm = 'ball_tree'

        # NN Test Consts
        self.NNT_neurons_list = np.linspace(10, 120, 15).astype(int)
        self.NNT_layers_list = np.linspace(1, 6, 6).astype(int)
        self.NNT_alpha_list = np.linspace(1e-6, 1e-4, 5)

        # NN Learner Consts
        self.NNL_alpha = 1e-5
        self.NNL_n_nodes = 50
        self.NNL_n_layers = 3

        # SVM Tests Consts
        self.SVMT_kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        self.SVMT_c_list = np.linspace(0.1, 3, 20)
        # SVM Learner Consts
        self.SVML_C = 1
        self.SVML_kernel = 'rbf'


class BaseTest(ABC):
    def __init__(self, details, name='', n_jobs=3, verbose=True):
        self.Name = name
        self.N_jobs = n_jobs

        self._details = details
        self._verbose = verbose
        self._learner = None

        self._scoring_metric = "f1_weighted"
        self._validation_fold_iterator = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

        # results of predictions
        self._predictions = None

    def run(self):
        """
        Just runs the basic training and testing. run_additional is called to do addition graphing and testing if need be
        :return:
        """
        if self._verbose:
            print("Starting " + self.Name + " tests.")
            print("      Starting timing tests.")
        train_time_ms, query_time_ms, prediction_scores = self.multi_run_train_test_timing()

        if self._verbose:
            print("      Starting Learning Curve. ")

        self.run_learning_curve()

        if self._verbose:
            print("      Starting Additional Tests.")

        # self.run_additional()

        # return train_time_ms, query_time_ms, prediction_scores

    def multi_run_train_test_timing(self, number_of_runs=1):
        """Make multiple runs to get more accurate timing for testing and training the model"""

        train_times = []
        query_times = []
        score_list = []

        for _ in range(0, number_of_runs):
            temp_train, temp_query, predictions = self.single_train_predict_with_timing()
            train_times.append(temp_train)
            query_times.append(temp_query)

            temp_score = f1_score(self._details.ds.test_y, predictions, average='weighted')

            score_list.append(temp_score)

        return train_times, query_times, score_list

    def single_train_predict_with_timing(self):
        ds = self._details.ds

        curr_time = datetime.datetime.now()
        self._learner.train(ds.train_x, ds.train_y)
        train_time = datetime.datetime.now() - curr_time

        curr_time = datetime.datetime.now()
        predictions = self._learner.test(ds.test_x)
        query_time = datetime.datetime.now() - curr_time

        return train_time.microseconds, query_time.microseconds, predictions

    def run_learning_curve(self, train_sizes=np.linspace(0.1, 1.0, 10)):

        # TODO: Should I be using test or train set here?
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            self._learner.Classifier,
            self._details.ds.train_x,
            self._details.ds.train_y,
            scoring="f1_weighted",
            cv=self._validation_fold_iterator,
            n_jobs=self.N_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

        fit_times_ms = fit_times * 1000
        plot_learning_curve(train_scores, test_scores, train_sizes, self.Name, folder=self._details.ds.name)
        plot_scalability(fit_times_ms, train_sizes, self.Name, folder=self._details.ds.name)

        return

    def run_multi_run_hyperparameter_validation(self, method, hyperparameter, n_runs=4, bar_chart=False):
        """

        :param method: should return a ndarray of hyperparameter keys, train scores, and test_scores
        :return:
        """

        train_scores = []
        test_scores = []
        param_list = None

        for _ in range(0, n_runs):
            param_list, temp_train_scores, temp_test_scores = method()

            train_scores.append(temp_train_scores)
            test_scores.append(temp_test_scores)

        train_scores_np = np.concatenate(train_scores, axis=1)
        test_scores_np = np.concatenate(test_scores, axis=1)

        if bar_chart:
            plot_hyperparam_validation_bar_chart(train_scores_np, test_scores_np, param_list, self.Name, hyperparameter, folder=self._details.ds.name)
        else:
            plot_hyperparam_validation_curve(train_scores_np, test_scores_np, param_list, self.Name, hyperparameter, folder=self._details.ds.name)

        if self._verbose:
            print("      " + hyperparameter + " validation complete.")
        return

    @abstractmethod
    def run_additional(self):
        """
        run_additional is called to do addition graphing and testing if need be
        :param predictions: results of the testing phase of the run_basic() routine.
        :return:
        """
        pass
