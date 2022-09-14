from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.SVMLearner import SVMLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve, plot_hyperparam_validation_bar_chart


class SVMTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='SVM')
        self._learner = SVMLearner()

    def run_additional(self):
        self.run_hyperparameter_validation()
        self.run_various_kernels()
        pass

    def run_various_kernels(self):
        cv = ShuffleSplit(n_splits=10, test_size=0.2)

        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
        train_scores = []
        test_scores = []

        for kernel in kernel_list:
            temp_learner = SVMLearner(kernel=kernel)
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


        plot_hyperparam_validation_bar_chart(train_scores, test_scores, kernel_list, self.Name, 'Kernel')
        return

    def run_hyperparameter_validation(self):
        self.run_c_validation()
        return

    def run_c_validation(self):
        """ Evaluate Regularization parameter """
        cv = ShuffleSplit(n_splits=10, test_size=0.2)

        c_list = np.linspace(0.1, 3, 30)
        train_scores = []
        test_scores = []

        for c in c_list:
            temp_learner = SVMLearner(C=c)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, c_list, self.Name, 'C')
