import numpy as np
import pandas as pd
import mlrose_hiive
import matplotlib.pyplot as plt
import time
import datetime
import os

from mlrose_hiive.runners import SKMLPRunner, NNGSRunner

from util import loading_data
from util.graphing import plot_learning_curve, plot_scalability, plot_loss_curves
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate, KFold


class NNAlgo(object):
    def __init__(self, backprop_algo='random_hill_climb', short_name='rhc', learning_rate=1e-5, max_iterations=3000,
                 grid_search_params=None, default_params=None):
        if grid_search_params is None:
            grid_search_params = {}
        if default_params is None:
            default_params = {}

        self.NNStructure = [30]
        self.LearningRate = learning_rate
        self.BackpropAlgo = backprop_algo
        self.MaxIterations = max_iterations
        self.ShortName = short_name

        base_default_grid_search_params = {
            'max_iters': [2000],
            'learning_rate_init': [0.1, 0.2, 0.4, 0.8],
            'learning_rate': [1e-5, 1e-3, 1e-2],
            'hidden_layer_sizes': [[30]],
            'activation': [mlrose_hiive.neural.activation.relu],
        }

        base_default_params = {
            'seed': 123456,
            'iteration_list': 2 ** np.arange(13),
            'max_attempts': 500,
            'n_jobs': 7,
            'cv': 5,
        }

        self.GridSearchParams = {**base_default_grid_search_params, **grid_search_params}
        self.Params = {**base_default_params, **default_params}
        return

    def construct_learner(self):
        return mlrose_hiive.NeuralNetwork(hidden_nodes=self.NNStructure,
                                          algorithm=self.BackpropAlgo,
                                          max_iters=self.MaxIterations,
                                          learning_rate=self.LearningRate,
                                          is_classifier=True,
                                          curve=True)

    def run_grid_search(self, ds):
        experiment_name = 'nngs_' + self.BackpropAlgo
        cx_skr = NNGSRunner(x_train=ds.train_x, y_train=ds.train_y,
                            x_test=ds.test_x, y_test=ds.test_y,
                            algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                            experiment_name=experiment_name,
                            grid_search_parameters=self.GridSearchParams,
                            **self.Params)

        run_stats_df, curves_df, cv_results_df, cx_sr = cx_skr.run()
        return run_stats_df, curves_df, cv_results_df, cx_sr

    def run_skmlp_grid_search(self, ds):
        skmlp_grid_search_parameters = {
            **self.DefaultGridSearchParams,
            'max_iters': [5000],
            'learning_rate_init': [0.0001],
            'activation': [mlrose_hiive.neural.activation.sigmoid],
        }

        skmlp_default_parameters = {
            **self.DefaultParams,
            'early_stopping': True,
            'tol': 1e-05,
            'alpha': 0.001,
            'solver': 'lbfgs',
        }

        cx_skr = SKMLPRunner(x_train=ds.train_x, y_train=ds.train_y,
                             x_test=ds.test_x, y_test=ds.test_y,
                             experiment_name='skmlp_clean',
                             grid_search_parameters=skmlp_grid_search_parameters,
                             **skmlp_default_parameters)

        run_stats_df, curves_df, cv_results_df, cx_sr = cx_skr.run()
        return


class NNTester(object):
    def __init__(self, n_jobs=7):
        self.N_Jobs = n_jobs
        self.Builder = NNBuilder()
        self.BackpropAlgos = []

        self._construct_algos()
        return

    def _construct_algos(self):
        self.BackpropAlgos.append(self.Builder.build_gradient_descent())
        self.BackpropAlgos.append(self.Builder.build_rhc())
        self.BackpropAlgos.append(self.Builder.build_sa())
        self.BackpropAlgos.append(self.Builder.build_ga())
        return

    def run(self):
        seed = 123456
        print("Seed {}".format(seed))

        print("Load datasets\r\n")

        ds_red_wine = loading_data.load_red_wine()
        ds_diabetic = loading_data.load_diabetic()

        print("Starting Tests....")

        self.exp_loss_curve(ds_red_wine)
        self.exp_scalability(ds_red_wine)

        self.exp_loss_curve(ds_diabetic)
        self.exp_scalability(ds_diabetic)
        return

    def _building_my_own_cv(self, ds):
        """I hope it doens't come to this"""
        sss = StratifiedShuffleSplit(n_splits=7, test_size=0.1, random_state=0)

        for train_index, test_index in sss.split(ds.train_x, ds.train_y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = ds.train_x[train_index], ds.train_x[test_index]
            y_train, y_test = ds.train_y[train_index], ds.train_y[test_index]

        return

    def _get_single_algo_loss_curve(self, ds, algo):
        loss_curves = []

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

        temp_learner = algo.construct_learner()

        res = cross_validate(
            temp_learner,
            ds.train_x,
            ds.train_y,
            scoring="accuracy",
            cv=sss,
            n_jobs=self.N_Jobs,
            return_train_score=True,
            return_estimator=True
        )

        for e in res['estimator']:
            loss_curves.append(e.fitness_curve)

        return loss_curves

    def exp_loss_curve(self, ds):
        loss_curves_dict = {
            'rhc': [],
            'sa': [],
            'ga': [],
            'gd': [],
        }

        for algo in self.BackpropAlgos:
            print("Generating Loss Curves for ", algo.BackpropAlgo)
            loss_curves = self._get_single_algo_loss_curve(ds, algo)
            loss_curves_dict[algo.ShortName] = loss_curves

        plot_loss_curves(loss_curves_dict, ds.name)
        return


    def exp_scalability(self, ds, backprop_algo='gradient_descent'):
        learner = mlrose_hiive.NeuralNetwork(hidden_nodes=self.NNStructure, algorithm=backprop_algo,
                                             max_iters=self.MaxIterations, learning_rate=self.LearningRate, curve=True)
        train_sizes = np.linspace(0.1, 1.0, 10)
        validation_fold_iterator = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            learner,
            ds.train_x,
            ds.train_y,
            scoring="accuracy",
            cv=validation_fold_iterator,
            n_jobs=self.N_Jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

        fit_times_ms = fit_times * 1000

        plot_learning_curve(train_scores, test_scores, train_sizes, backprop_algo, folder='nn_' + ds.name)
        plot_scalability(fit_times_ms, train_sizes, backprop_algo, folder='nn_' + ds.name)
        return


class NNBuilder(object):
    def build_gradient_descent(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'max_iters': [2000],
            'learning_rate_init': [0.3, 0.4, 0.5],
            'learning_rate': [1e-6, 1e-5, 1e-4],
        }

        default_params = {
            'seed': 123456,
            'iteration_list': 2 ** np.arange(13),
            'max_attempts': 500,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='gradient_descent',
                      short_name='gd',
                      learning_rate=1e-3,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params)

        return algo

    def build_rhc(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'max_iters': [2000],
            'learning_rate_init': [0.1, 0.4, 0.8],
            'learning_rate': [1e-5, 1e-3, 1e-2],
        }

        default_params = {
            'seed': 123456,
            'iteration_list': 2 ** np.arange(13),
            'max_attempts': 500,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='random_hill_climb',
                      short_name='rhc',
                      learning_rate=1e-5,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params)

        return algo

    def build_sa(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'max_iters': [2000],
            'learning_rate': [1e-5, 1e-3, 1e-2],
            'schedule': [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
        }

        default_params = {
            'seed': 123456,
            'iteration_list': 2 ** np.arange(13),
            'max_attempts': 500,
            'n_jobs': 7,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='simulated_annealing',
                      short_name='sa',
                      learning_rate=1e-5,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params)

        return algo

    def build_ga(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'max_iters': [2000],
            'learning_rate': [1e-5, 1e-3, 1e-2],
            'mutation_prob': [0.05, 0.1, 0.2],
            'pop_size': [50, 200, 400],
        }

        default_params = {
            'seed': 123456,
            'iteration_list': 2 ** np.arange(13),
            'max_attempts': 500,
            'n_jobs': 7,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='genetic_alg',
                      short_name='ga',
                      learning_rate=1e-5,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params)

        return algo


class NNGridSearchExecutor(object):
    """ Building this to handle all of the NN object creations and gridsearch parameters"""

    def __init__(self, ds):
        self.Dataset = ds
        self.Builder = NNBuilder()
        return

    def run_all_grid_searches(self):
        self.run_gd_gs()
        self.run_rhc_gs()
        self.run_sa_gs()
        self.run_ga_gs()
        return

    def run_rhc_gs(self):
        print('Running RHC GridSearch')
        algo = self.Builder.build_rhc()
        res_tuple = algo.run_grid_search(self.Dataset)
        self._print_results(res_tuple)
        return

    def run_sa_gs(self):
        print('Running SA GridSearch')
        algo = self.Builder.build_sa()
        res_tuple = algo.run_grid_search(self.Dataset)
        self._print_results(res_tuple)
        return

    def run_ga_gs(self):
        print('Running GA GridSearch')
        algo = self.Builder.build_ga()
        res_tuple = algo.run_grid_search(self.Dataset)
        self._print_results(res_tuple)
        return

    def run_gd_gs(self):
        print('Running GradDesc GridSearch')
        algo = self.Builder.build_gradient_descent()
        res_tuple = algo.run_grid_search(self.Dataset)
        self._print_results(res_tuple)
        return

    def _print_results(self, res_tuple):
        run_stats_df, curves_df, cv_results_df, cx_sr = res_tuple
        print('Best Score: ', cx_sr.best_score_)
        print('Best Params: ')
        print(cx_sr.best_params_)
        return


if __name__ == "__main__":
    print("Load datasets\r\n")

    ds_red_wine = loading_data.load_red_wine()
    ds_diabetic = loading_data.load_diabetic()

    print("Starting Tests....")

    tester = NNTester()
    tester.run()

    # Run GS on NN algos to try to get somekind of results
    rw_grid_search = NNGridSearchExecutor(ds_red_wine)
    rw_grid_search.run_all_grid_searches()
