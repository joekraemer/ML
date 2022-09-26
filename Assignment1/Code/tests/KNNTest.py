from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.KNNLearner import KNNLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class KNNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='KNN')
        self._learner = KNNLearner(n_neighbors=self._details.KNNL_n_neighbors, algorithm=self._details.KNNL_algorithm)

        self.algo_list = self._details.KNNT_algo_list
        self.k_list = self._details.KNNT_k_list

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass


    def run_hyperparameter_validation(self):
        self.run_multi_run_hyperparameter_validation(self.run_algo_validation, 'algo_multi', bar_chart=True)
        self.run_multi_run_hyperparameter_validation(self.run_k_validation, 'k_multi')
        return

    def run_algo_validation(self):
        """ Evaluate algo Hyperparameters """

        train_scores = []
        test_scores = []

        for algo in self.algo_list:
            temp_learner = KNNLearner(n_neighbors=self._details.KNNL_n_neighbors, algorithm=algo)
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

        return self.algo_list, np.asarray(train_scores), np.asarray(test_scores)

    def run_k_validation(self):
        # for KNN we will evaluate k and TODO

        """ Evaluate Leaf Size """
        train_scores = []
        test_scores = []

        for k in self.k_list:
            temp_learner = KNNLearner(n_neighbors=k, algorithm=self._details.KNNL_algorithm)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, self.k_list, self.Name, 'N_Neighbors', folder=self._details.ds.name)
        return self.k_list, np.asarray(train_scores), np.asarray(test_scores)
