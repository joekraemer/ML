import numpy as np
from tests.BaseTest import BaseTest
import datetime

from learners.DecisionTreeLearner import DecisionTreeLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve


class DecisionTreeTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Decision Tree')
        self._learner = DecisionTreeLearner(leaf_size=details.DTT_leaf_size, ccp_alpha=details.DTT_ccp_alpha, max_depth=details.DTT_max_depth)

        self.leaf_sizes_list = details.DTT_leaf_sizes_list
        self.ccp_alpha_list = details.DTT_ccp_alpha_list
        self.max_depth_list = details.DTT_max_depth_list

    def run_additional(self):
        self.run_hyperparameter_validation()
        pass

    def run_hyperparameter_validation(self):
        self.run_leaf_size_validation()
        self.run_ccp_alpha_validation()
        self.run_max_depth_validation()
        return

    def run_leaf_size_validation(self):
        # for Decision Trees we will evaluate leaf size and TODO

        """ Evaluate Leaf Size """

        train_scores = []
        test_scores = []

        for leaf_size in self.leaf_sizes_list:
            temp_learner = DecisionTreeLearner(leaf_size=leaf_size, ccp_alpha=self._details.DTT_ccp_alpha, max_depth=self._details.DTT_max_depth)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, self.leaf_sizes_list, self.Name, 'Leaf Size', folder=self._details.ds.name, enable_dots=False)

    def run_ccp_alpha_validation(self):
        """  evaluate Cost-Complexity Pruning """
        train_scores = []
        test_scores = []

        for a in self.ccp_alpha_list:
            temp_learner = DecisionTreeLearner(ccp_alpha=a)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, self.ccp_alpha_list, self.Name, 'CCP Alpha', folder=self._details.ds.name)


    def run_max_depth_validation(self):
        """  evaluate max_depth """
        train_scores = []
        test_scores = []

        for d in self.max_depth_list:
            temp_learner = DecisionTreeLearner(max_depth=d)
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

        plot_hyperparam_validation_curve(train_scores, test_scores, self.max_depth_list, self.Name, 'Max Depth', folder=self._details.ds.name)
