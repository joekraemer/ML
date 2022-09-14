from abc import ABC, abstractmethod
import datetime
import numpy as np

from sklearn.model_selection import learning_curve, ShuffleSplit

from util.graphing import plot_learning_curve, plot_scalability
from util.graphing import plot_hyperparam_validation_curve


class TestDetails(object):
    def __init__(self, ds, seed):
        self.ds = ds
        self.seed = seed


class BaseTest(ABC):
    def __init__(self, details, name='', n_jobs=1, verbose=False):
        self.Name = name
        self.N_jobs = n_jobs

        self._details = details
        self._verbose = verbose
        self._learner = None

        # results of predictions
        self._predictions = None

    def run(self):
        """
        Just runs the basic training and testing. run_additional is called to do addition graphing and testing if need be
        :return:
        """

        train_time_ms, query_time_ms = self.multi_run_train_test_timing()
        self.run_additional()
        self.run_learning_curve()

        return train_time_ms, query_time_ms

    def multi_run_train_test_timing(self, number_of_runs=10):
        """Make multiple runs to get more accurate timing for testing and training the model"""

        train_times = []
        query_times = []
        for _ in range(0, number_of_runs):
            temp_train, temp_query, _ = self.single_train_predict_with_timiing()
            train_times.append(temp_train)
            query_times.append(temp_query)

        return train_times, query_times

    def single_train_predict_with_timiing(self):
        ds = self._details.ds

        curr_time = datetime.datetime.now()
        self._learner.train(ds.train_x, ds.train_y)
        train_time = datetime.datetime.now() - curr_time

        curr_time = datetime.datetime.now()
        predictions = self._learner.test(ds.test_x)
        query_time = datetime.datetime.now() - curr_time

        return train_time.microseconds, query_time.microseconds, predictions

    def run_learning_curve(self, train_sizes=np.linspace(0.1, 1.0, 10)):

        cv = ShuffleSplit(n_splits=20, test_size=0.1, random_state=0)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            self._learner.Classifier,
            self._details.ds.test_x,
            self._details.ds.test_y,
            scoring="accuracy",
            cv=cv,
            n_jobs=self.N_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

        plot_learning_curve(train_scores, test_scores, train_sizes, self.Name)
        plot_scalability(fit_times, train_sizes, self.Name)

        return

    def run_multi_run_hyperparameter_validation(self, method, hyperparameter, n_runs=4):
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

        plot_hyperparam_validation_curve(train_scores_np, test_scores_np, param_list, self.Name, hyperparameter)
        return

    @abstractmethod
    def run_additional(self):
        """
        run_additional is called to do addition graphing and testing if need be
        :param predictions: results of the testing phase of the run_basic() routine.
        :return:
        """
        pass
