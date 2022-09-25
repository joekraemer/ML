from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.BoostedLearner import BoostedLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class BoostedTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Boosted')
        self._learner = BoostedLearner(n_estimators=details.boost_n_estimators, learning_rate=details.boost_learning_rate)

        # Constants
        self.n_estimators_list = details.boost_n_estimators_list
        self.lr_list = details.boost_lr_list

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        # for NN we will evaluate number of neurons and number of layers

        self.run_multi_run_hyperparameter_validation(self.run_lr_validation, 'lr_multi')
        self.run_multi_run_hyperparameter_validation(self.run_num_estimators_validation, 'num_estimators_multi')
        return

    def run_num_estimators_validation(self):
        """ Evaluate Hyperparameters """

        train_scores = []
        test_scores = []

        for n in self.n_estimators_list:
            temp_learner = BoostedLearner(n_estimators=n, learning_rate=self._details.boost_learning_rate)
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

        return self.n_estimators_list, np.asarray(train_scores), np.asarray(test_scores)

    def run_lr_validation(self):
        """ Evaluate Hyperparameters """

        train_scores = []
        test_scores = []

        for lr in self.lr_list:
            temp_learner = BoostedLearner(learning_rate=lr, n_estimators=self._details.boost_n_estimators)
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

        return self.lr_list, np.asarray(train_scores), np.asarray(test_scores)
