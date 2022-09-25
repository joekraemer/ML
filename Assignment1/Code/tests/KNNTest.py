from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.KNNLearner import KNNLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class KNNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='KNN')
        self._learner = KNNLearner()

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass


    def run_hyperparameter_validation(self):
        self.run_multi_run_hyperparameter_validation(self.run_algo_validation, 'algo_multi', bar_chart=True)
        self.run_multi_run_hyperparameter_validation(self.run_k_validation, 'k_multi')
        return

    def run_algo_validation(self):
        """ Evaluate algo Hyperparameters """

        algo_list = ['ball_tree', 'kd_tree', 'brute']

        train_scores = []
        test_scores = []

        for algo in algo_list:
            temp_learner = KNNLearner(n_neighbors=15, algorithm=algo)
            res = cross_validate(
                temp_learner.Classifier,
                self._details.ds.train_x,
                self._details.ds.train_y,
                scoring=self._scoring_metric,
                cv=self._validation_fold_iterator,
                n_jobs=self.N_jobs,
                return_train_score=True
            )

            train_scores.append(res['train_score'])
            test_scores.append(res['test_score'])

        return algo_list, np.asarray(train_scores), np.asarray(test_scores)

    def run_k_validation(self):
        # for KNN we will evaluate k and TODO

        """ Evaluate Leaf Size """
        k_list = np.linspace(1, 30, 20).astype(int)
        train_scores = []
        test_scores = []

        for k in k_list:
            temp_learner = KNNLearner(n_neighbors=k)
            res = cross_validate(
                temp_learner.Classifier,
                self._details.ds.train_x,
                self._details.ds.train_y,
                scoring=self._scoring_metric,
                cv=self._validation_fold_iterator,
                n_jobs=self.N_jobs,
                return_train_score=True
            )

            train_scores.append(res['train_score'])
            test_scores.append(res['test_score'])

        plot_hyperparam_validation_curve(train_scores, test_scores, k_list, self.Name, '# Neighbors', folder=self._details.ds.name)
        return k_list, np.asarray(train_scores), np.asarray(test_scores)
