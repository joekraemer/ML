import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from tests.BaseTest import BaseTest
import datetime

from learners.DecisionTreeLearner import DecisionTreeLearner
from sklearn.model_selection import cross_validate, ShuffleSplit

from util.graphing import plot_hyperparam_validation_curve, plot_helper


class DecisionTreeTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Decision Tree')
        self._learner = DecisionTreeLearner(leaf_size=details.DTL_leaf_size, ccp_alpha=details.DTL_ccp_alpha, max_depth=details.DTL_max_depth)

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
        self.ccp_alpha_investigation()
        return

    def run_leaf_size_validation(self):
        # for Decision Trees we will evaluate leaf size and TODO

        """ Evaluate Leaf Size """

        train_scores = []
        test_scores = []

        for leaf_size in self.leaf_sizes_list:
            temp_learner = DecisionTreeLearner(leaf_size=leaf_size, ccp_alpha=0, max_depth=None)
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
            temp_learner = DecisionTreeLearner(ccp_alpha=a, leaf_size=1, max_depth=None)
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
            temp_learner = DecisionTreeLearner(max_depth=d, leaf_size=self._details.DTL_leaf_size, ccp_alpha=0)
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

    def ccp_alpha_investigation(self):

        clf = DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(self._details.ds.train_x, self._details.ds.train_y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        plot_helper('Total Impurity vs effective alpha for training set', 'ccp_alphas_impurity_vs_alpha', self._details.ds.name, show=False)

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(self._details.ds.train_x, self._details.ds.train_y)
            clfs.append(clf)
        print(
            "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                clfs[-1].tree_.node_count, ccp_alphas[-1]
            )
        )

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()
        plot_helper('', 'depth_vs_alpha', self._details.ds.name, show=False)


        train_scores = [clf.score(self._details.ds.train_x, self._details.ds.train_y) for clf in clfs]
        test_scores = [clf.score(self._details.ds.test_x, self._details.ds.test_y) for clf in clfs]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()

        plot_helper('Accuracy vs alpha for training and testing sets', 'ccp_alphas_accuracy_vs_alpha', self._details.ds.name, show=False)
        return
