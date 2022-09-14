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
        # for KNN we will evaluate k and TODO

        """ Evaluate Leaf Size """
        # TODO need to see if this CV splits the same way for each cross_validation
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        k_list = np.linspace(1, 30, 15).astype(int)
        train_scores = []
        test_scores = []

        for k in k_list:
            temp_learner = KNNLearner(n_neighbors=k)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, k_list, self.Name, '# Neighbors')

