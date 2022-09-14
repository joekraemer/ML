import numpy as np
from tests.BaseTest import BaseTest
import datetime

from learners.DecisionTreeLearner import DecisionTreeLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class DecisionTreeTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Decision Tree')
        self._learner = DecisionTreeLearner()

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        # for Decision Trees we will evaluate leaf size and TODO

        """ Evaluate Leaf Size """
        # TODO need to see if this CV splits the same way for each cross_validation
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

        leaf_sizes_list = np.linspace(1, 60, 10).astype(int)
        train_scores = []
        test_scores = []

        for leaf_size in leaf_sizes_list:
            temp_learner = DecisionTreeLearner(leaf_size=leaf_size)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, leaf_sizes_list, self.Name, 'Leaf Size')
