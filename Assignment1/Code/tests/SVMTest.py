from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.SVMLearner import SVMLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve, plot_hyperparam_validation_bar_chart


class SVMTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='SVM')
        self._learner = SVMLearner(C=self._details.SVML_C, kernel=self._details.SVML_kernel)

        self.kernel_list = self._details.SVMT_kernel_list
        self.c_list = self._details.SVMT_c_list

    def run_additional(self):
        self.run_hyperparameter_validation()
        self.run_various_kernels()
        pass

    def run_various_kernels(self):

        train_scores = []
        test_scores = []

        for kernel in self.kernel_list:
            temp_learner = SVMLearner(kernel=kernel, C=self._details.SVML_C)
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


        plot_hyperparam_validation_bar_chart(train_scores, test_scores, self.kernel_list, self.Name, 'Kernel', folder=self._details.ds.name)
        return

    def run_hyperparameter_validation(self):
        self.run_c_validation()
        return

    def run_c_validation(self):
        """ Evaluate Regularization parameter """

        train_scores = []
        test_scores = []

        for c in self.c_list:
            temp_learner = SVMLearner(C=c, kernel=self._details.SVML_kernel)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, self.c_list, self.Name, 'C', folder=self._details.ds.name)
