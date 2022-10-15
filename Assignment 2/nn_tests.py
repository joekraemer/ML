import numpy as np
import pandas as pd
import mlrose_hiive
import matplotlib.pyplot as plt
import time
import datetime
import os

from mlrose_hiive.runners import SKMLPRunner, NNGSRunner

from util import loading_data
from util.logging_tables import logging_scoring_metrics
from util.graphing import plot_learning_curve, plot_scalability, plot_loss_curves
from sklearn.metrics import f1_score, accuracy_score, classification_report

from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate, KFold


class NNAlgo(object):
    def __init__(self, backprop_algo='random_hill_climb', short_name='rhc', full_algo=mlrose_hiive.algorithms.rhc.random_hill_climb , learning_rate=1e-5, max_iterations=3000,
                 grid_search_params=None, default_params=None, debug=False):
        if grid_search_params is None:
            grid_search_params = {}
        if default_params is None:
            default_params = {}

        if debug:
            self.NNStructure = [40, 40]
        else:
            self.NNStructure = [50, 50, 50]

        self.FullAlgo = full_algo
        self.LearningRate = learning_rate
        self.BackpropAlgo = backprop_algo
        self.MaxIterations = max_iterations
        self.ShortName = short_name

        base_default_grid_search_params = {
            'max_iters': [5000],
            'learning_rate_init': [0.5],
            'learning_rate': [1e-5, 1e-4, 1e-2],
            'hidden_layer_sizes': [self.NNStructure],
            'activation': [mlrose_hiive.neural.activation.relu],
        }

        base_default_params = {
            'seed': 123456,
            'iteration_list': [5000],
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
                            algorithm=self.FullAlgo,
                            experiment_name=experiment_name,
                            grid_search_parameters=self.GridSearchParams,
                            **self.Params)

        run_stats_df, curves_df, cv_results_df, cx_sr = cx_skr.run()

        y_pred = cx_sr.predict(ds.test_x)
        y_pred_train = cx_sr.predict(ds.train_x)
        test = [classification_report(ds.test_y, y_pred)]
        train = [classification_report(ds.train_y, y_pred_train)]
        logging_scoring_metrics(test, folder='nn_' + ds.name, name=self.ShortName + '_test')
        logging_scoring_metrics(train, folder='nn_' + ds.name, name=self.ShortName + '_train')

        return run_stats_df, curves_df, cv_results_df, cx_sr


class NNTester(object):
    def __init__(self, n_jobs=7, n_runs=3, debug=True):
        self.Debug = debug
        self.N_Jobs = n_jobs
        self.Builder = NNBuilder(self.Debug)
        self.BackpropAlgos = []
        self.N_Runs = n_runs

        if self.Debug:
            self.N_Runs = 1
            n_splits = 1
        else:
            n_splits = 3

        self.SSS = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)

        self._construct_algos()
        return

    def _construct_algos(self):
        self.BackpropAlgos.append(self.Builder.build_gradient_descent())
        self.BackpropAlgos.append(self.Builder.build_rhc())
        self.BackpropAlgos.append(self.Builder.build_sa())
        self.BackpropAlgos.append(self.Builder.build_ga())
        return

    def run(self, ds):
        seed = 123456
        print("Seed {}".format(seed))

        print("Starting Tests....")

        self.exp_loss_curve(ds_red_wine)
        self.exp_learning_curve_all_algos(ds)
        return

    def _get_single_algo_loss_curve(self, ds, algo):
        loss_curves = []

        temp_learner = algo.construct_learner()

        res = cross_validate(
            temp_learner,
            ds.train_x,
            ds.train_y,
            scoring="accuracy",
            cv=self.SSS,
            n_jobs=self.N_Jobs,
            return_train_score=True,
            return_estimator=True
        )

        for e in res['estimator']:
            shape = e.fitness_curve.shape

            # Only GD returns 1D array fitness curves
            if len(shape) > 1:
                lc = e.fitness_curve[:, 0]

            else:
                lc = e.fitness_curve
                if lc[0] < 0:
                    # GD returns loss that is negative
                    lc = lc * -1

            loss_curves.append(lc)

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

    def exp_learning_curve_all_algos(self, ds):
        for algo in self.BackpropAlgos:
            self.exp_learning_curve(ds, algo)

    def exp_learning_curve(self, ds, algo):
        train_sizes = np.linspace(0.1, 1.0, 4)

        temp_learner = algo.construct_learner()

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            temp_learner,
            ds.train_x,
            ds.train_y,
            scoring="f1_weighted",
            cv=self.SSS,
            n_jobs=self.N_Jobs,
            train_sizes=train_sizes,
            return_times=True,
        )

        fit_times_ms = fit_times * 1000

        plot_learning_curve(train_scores, test_scores, train_sizes, algo.ShortName, folder='nn_' + ds.name)
        plot_scalability(fit_times_ms, train_sizes, algo.ShortName, folder='nn_' + ds.name)
        return


class NNBuilder(object):
    def __init__(self, debug=False):
        self.Debug = debug

    def build_gradient_descent(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'learning_rate_init': [0.5],
            'learning_rate': [1e-5, 1e-4, 1e-2],
        }

        default_params = {
            'seed': 123456,
            'max_attempts': 500,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='gradient_descent',
                      full_algo=mlrose_hiive.algorithms.gd.gradient_descent,
                      short_name='gd',
                      learning_rate=1e-3,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params,
                      debug=self.Debug)

        return algo

    def build_rhc(self):
        grid_search_params = {
            'learning_rate': [1e-5, 1e-3, 1e-2],
            'restarts': [0, 5, 10],
        }

        default_params = {
            'seed': 123456,
            'max_attempts': 500,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='random_hill_climb',
                      full_algo=mlrose_hiive.algorithms.rhc.random_hill_climb,
                      short_name='rhc',
                      learning_rate=1e-5,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params,
                      debug=self.Debug)

        return algo

    def build_sa(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'learning_rate': [1e-5, 1e-3, 1e-2],
            'schedule': [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
        }

        default_params = {
            'seed': 123456,
            'max_attempts': 500,
            'n_jobs': 7,
            'cv': 5,
        }

        algo = NNAlgo(backprop_algo='simulated_annealing',
                      full_algo=mlrose_hiive.algorithms.sa.simulated_annealing,
                      short_name='sa',
                      learning_rate=1e-5,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params,
                      debug=self.Debug)

        return algo

    def build_ga(self):
        # 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'

        grid_search_params = {
            'learning_rate': [1e-4, 1e-2, 1e-1],
            'mutation_prob': [0.05, 0.1, 0.3],
            'pop_size': [50, 100, 200],
        }

        default_params = {
            'seed': 123456,
            'max_attempts': 200,
            'n_jobs': 7,
            'cv': 5,
            'pop_size': 150,
            'mutation_prob': 0.2,
        }

        algo = NNAlgo(backprop_algo='genetic_alg',
                      short_name='ga',
                      learning_rate=1e-2,
                      max_iterations=2000,
                      grid_search_params=grid_search_params,
                      default_params=default_params,
                      debug=self.Debug)

        return algo


class NNGridSearchExecutor(object):
    """ Building this to handle all of the NN object creations and gridsearch parameters"""

    def __init__(self, ds, debug=False):
        self.Dataset = ds
        self.Builder = NNBuilder(debug=debug)
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

    # Run GS on NN algos to try to get somekind of results
    rw_grid_search = NNGridSearchExecutor(ds_red_wine, debug=False)
    rw_grid_search.run_all_grid_searches()

    tester = NNTester(n_jobs=7, debug=True)
    tester.run(ds_red_wine)


