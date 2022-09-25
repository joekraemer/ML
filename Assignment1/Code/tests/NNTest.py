import pandas as pd
from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.NNLearner import NNLearner
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit

from util.graphing import plot_hyperparam_validation_curve, plot_single_curve


class NNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='NN')
        self._learner = NNLearner()

    def run_additional(self):
        self.run_loss_curve()
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        # for NN we will evaluate number of neurons and number of layers
        # self.run_neurons_validation()
        # self.run_alpha_validation()
        # self.run_layers_validation()
        self.run_loss_curve()
        return

    def run_neurons_validation(self):
        """ Evaluate Hyperparameters """

        neurons_list = np.linspace(5, 150, 20).astype(int)
        train_scores = []
        test_scores = []

        for n in neurons_list:
            temp_learner = NNLearner(n_nodes=n)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, neurons_list, self.Name, 'Number of Nodes', folder=self._details.ds.name)

    def run_layers_validation(self):
        """ Evaluate layers Hyperparameters """

        layers_list = np.linspace(1, 6, 6).astype(int)
        train_scores = []
        test_scores = []

        for n in layers_list:
            temp_learner = NNLearner(n_layers=n)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, layers_list, self.Name, 'Number of Hidden Layers', folder=self._details.ds.name)

    def run_alpha_validation(self):
        """ Evaluate Hyperparameters """

        alpha_list = np.linspace(1e-6, 1e-4, 5)
        train_scores = []
        test_scores = []

        for a in alpha_list:
            temp_learner = NNLearner(alpha=a)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, alpha_list, self.Name, 'alpha', folder=self._details.ds.name)

    def run_loss_curve(self):

        loss_curves = []

        # Don't want to use the default iterator because it doesn't preserve the loss curve for each fold
        fold_iterator = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

        for _ in range(0, 9):
            temp_learner = NNLearner()
            res = cross_validate(
                temp_learner.Classifier,
                self._details.ds.train_x,
                self._details.ds.train_y,
                scoring=self._scoring_metric,
                cv=fold_iterator,
                n_jobs=self.N_jobs,
                return_train_score=True,
                return_estimator=True
            )

        for e in res['estimator']:
            loss_curves.append(e.loss_curve_)

        loss_curves_df = pd.DataFrame(loss_curves)

        plot_single_curve(loss_curves_df, 'NN_loss_curve', 'Loss Curve for Neural Network', 'Loss', 'iteration', folder=self._details.ds.name)

