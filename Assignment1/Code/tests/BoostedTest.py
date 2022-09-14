from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.BoostedLearner import BoostedLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class BoostedTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Boosted')
        self._learner = BoostedLearner()

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        # for NN we will evaluate number of neurons and number of layers

        self.run_multi_run_hyperparameter_validation(self.run_lr_validation, 'lr')
        self.run_num_estimators_validation()
        self.run_lr_validation()
        return

    def run_num_estimators_validation(self):
        """ Evaluate Hyperparameters """
        # TODO need to see if this CV splits the same way for each cross_validation
        cv = ShuffleSplit(n_splits=10, test_size=0.2)

        n_estimators_list = np.linspace(1, 80, 40).astype(int)
        train_scores = []
        test_scores = []

        for n in n_estimators_list:
            temp_learner = BoostedLearner(n_estimators=n)
            res = cross_validate(
                temp_learner.Classifier,
                self._details.ds.train_x,
                self._details.ds.train_y,
                scoring="accuracy",
                cv=cv,
                n_jobs=self.N_jobs,
                return_train_score=True
            )

            train_scores.append(res['train_score'])
            test_scores.append(res['test_score'])

        plot_hyperparam_validation_curve(train_scores, test_scores, n_estimators_list, self.Name, 'Number of Estimators')

    def run_lr_validation(self):
        """ Evaluate Hyperparameters """
        cv = ShuffleSplit(n_splits=10, test_size=0.2)

        lr_list = np.linspace(0.01, 10, 20)
        train_scores = []
        test_scores = []

        for lr in lr_list:
            temp_learner = BoostedLearner(learning_rate=lr)
            res = cross_validate(
                temp_learner.Classifier,
                self._details.ds.train_x,
                self._details.ds.train_y,
                scoring="accuracy",
                cv=cv,
                n_jobs=self.N_jobs,
                return_train_score=True
            )

            train_scores.append(res['train_score'])
            test_scores.append(res['test_score'])

        return lr_list, np.asarray(train_scores), np.asarray(test_scores)

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

            train_scores.append(temp_test_scores)
            test_scores.append(temp_test_scores)

        train_scores_np = np.concatenate(train_scores, axis=1)
        test_scores_np = np.concatenate(test_scores, axis=1)

        plot_hyperparam_validation_curve(train_scores_np, test_scores_np, param_list, self.Name, hyperparameter)

