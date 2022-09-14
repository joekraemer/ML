from tests.BaseTest import BaseTest
import datetime
import numpy as np

from learners.NNLearner import NNLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class NNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='NN')
        self._learner = NNLearner()

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        # for NN we will evaluate number of neurons and number of layers

        self.run_neurons_validation()
        self.run_alpha_validation()

        return

    def run_neurons_validation(self):
        """ Evaluate Hyperparameters """
        # TODO need to see if this CV splits the same way for each cross_validation
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        neurons_list = np.linspace(5, 200, 10).astype(int)
        train_scores = []
        test_scores = []

        for n in neurons_list:
            temp_learner = NNLearner(n_nodes=n)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, neurons_list, self.Name, 'Number of Nodes')


    def run_alpha_validation(self):
        """ Evaluate Hyperparameters """
        # TODO need to see if this CV splits the same way for each cross_validation
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        alpha_list = np.linspace(1e-6, 1e-4, 5)
        train_scores = []
        test_scores = []

        for a in alpha_list:
            temp_learner = NNLearner(alpha=a)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, alpha_list, self.Name, 'alpha')

