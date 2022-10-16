import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper
from util.logging_tables import save_obj_as_pickle, log_hyper_table


class HyperParamResult(object):
    def __init__(self, fitness_dict, evals_dict, time_dict, dataset, param):
        self.TimeDict = time_dict
        self.FitnessDict = fitness_dict
        self.EvalsDict = evals_dict
        self.Dataset = dataset
        self.Param = param


class HyperTester(object):
    def __init__(self, problem_constructor, dataset, n_runs=7, config=None, aws=False):
        num_cores = multiprocessing.cpu_count()
        if aws:
            self.N_Jobs = num_cores
        else:
            self.N_Jobs = num_cores - 1
        self.Dataset = dataset
        self.N_Runs = n_runs
        self.ProblemConstructor = problem_constructor


        default_hyper_params = {'num_restarts': [0, 4, 8, 12, 16],
                                'ga_pop_size': [50, 100, 200, 350],
                                'ga_default_pop': 100,
                                'ga_mut_prob': [0.02, 0.05, 0.1, 0.2],
                                'mimic_pop_size': [100, 200, 300, 400],
                                'mimic_keep_percent': [0.05, 0.1, 0.2, 0.3]}

        if config is not None:
            self.Hyper = {**default_hyper_params, **config}
        else:
            self.Hyper = default_hyper_params

        return

    def generic_hyperparam_multi(self, function, params, legend_base_label, filename_base,
                                 dataset):
        fitness_curve_dict = {}
        time_dict = {}
        evals_curve_dict = {}

        for p in params:
            fitness_curve_dict[str(p)] = []
            evals_curve_dict[str(p)] = []
            time_dict[str(p)] = []

        inputs = tqdm(params)

        for _ in range(0, self.N_Runs):

            processed_list_all = Parallel(n_jobs=self.N_Jobs)(delayed(function)(i) for i in inputs)

            for res in processed_list_all:
                fitness_curve_dict[res[0]].append(res[1][:, 0])
                evals_curve_dict[res[0]].append(res[1][:, 1])
                time_dict[res[0]].append(res[2])

        full_res = HyperParamResult(fitness_dict=fitness_curve_dict, evals_dict=evals_curve_dict, time_dict=time_dict, dataset=dataset, param=filename_base)

        filename = dataset + '_hyperparam' + filename_base
        folder = dataset

        save_obj_as_pickle(full_res, folder, filename)
        log_hyper_table(evals_dict=evals_curve_dict, fitness_dict=fitness_curve_dict, times_dict=time_dict, folder=folder, name=filename_base)
        # plotting
        plot_hyperparam_dict_generic(fitness_curve_dict, label=legend_base_label)
        plot_helper('', filename, folder)

    def hyperparam_rhc(self):
        self.generic_hyperparam_multi(function=self.run_single_hyper_rhc, params=self.Hyper['num_restarts'],
                                      legend_base_label='# restarts = ', filename_base='_rhc_restarts',
                                      dataset=self.Dataset)

        print("Completed RHC hyper-parameter testing!")
        return

    def run_single_hyper_rhc(self, r):
        problem, init_state = self.ProblemConstructor()

        start = time.time()
        _, _, fitness_curve = mlrose_hiive.random_hill_climb(problem, restarts=r, max_attempts=1000,
                                                             max_iters=10000,
                                                             init_state=init_state, curve=True)
        end = time.time() - start
        return str(r), fitness_curve, end

    def hyperparam_sa(self):
        problem, init_state = self.ProblemConstructor()

        _, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                    max_attempts=1000, max_iters=100000,
                                                                    init_state=init_state, curve=True)

        _, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
                                                                    max_attempts=1000, max_iters=100000,
                                                                    init_state=init_state, curve=True)

        _, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ArithDecay(),
                                                                    max_attempts=1000, max_iters=100000,
                                                                    init_state=init_state, curve=True)
        print("Completed SA hyper-parameter testing!")

        plt.figure()
        plt.plot(fitness_curve_sa_1[:, 0], label='decay = Exp')
        plt.plot(fitness_curve_sa_2[:, 0], label='decay = Geom')
        plt.plot(fitness_curve_sa_3[:, 0], label='decay = Arith')

        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plot_helper('', self.Dataset + '_hyperparam_sa_decay', self.Dataset)
        return

    def hyperparam_ga(self):
        self.generic_hyperparam_multi(function=self.run_single_hyper_ga_pop, params=self.Hyper['ga_pop_size'],
                                      legend_base_label='Population = ',
                                      filename_base='_ga_pop_size', dataset=self.Dataset)
        print("Completed GA population size hyper-parameter")

        self.generic_hyperparam_multi(function=self.run_single_hyper_ga_mut, params=self.Hyper['ga_mut_prob'],
                                      legend_base_label='Mutation Prob = ',
                                      filename_base='_ga_mutation_prob', dataset=self.Dataset)
        print("Completed GA mutation prob testing")
        return

    def run_single_hyper_ga_pop(self, pop):
        problem, init_state = self.ProblemConstructor()
        start = time.time()

        _, _, fitness_curve = mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=pop,
                                                       max_attempts=1000,
                                                       max_iters=10000, curve=True)
        end = time.time() - start
        return str(pop), fitness_curve, end

    def run_single_hyper_ga_mut(self, mut):
        problem, init_state = self.ProblemConstructor()
        start = time.time()

        _, _, fitness_curve = mlrose_hiive.genetic_alg(problem, mutation_prob=mut,
                                                       pop_size=self.Hyper['ga_default_pop'],
                                                       max_attempts=1000,
                                                       max_iters=10000, curve=True)
        end = time.time() - start
        return str(mut), fitness_curve, end

    def hyperparam_mimic(self):
        self.generic_hyperparam_multi(function=self.run_single_hyper_mimic_pop, params=self.Hyper['mimic_pop_size'],
                                      legend_base_label='Population = ',
                                      filename_base='_mimic_pop_size', dataset=self.Dataset)
        print("Completed MIMIC population size hyper-parameter")

        self.generic_hyperparam_multi(function=self.run_single_hyper_mimic_keep,
                                      params=self.Hyper['mimic_keep_percent'],
                                      legend_base_label='Keep % = ',
                                      filename_base='_mimic_keep_percent', dataset=self.Dataset)
        print("Completed MIMIC keep % testing")

    def run_single_hyper_mimic_pop(self, pop):
        problem, init_state = self.ProblemConstructor()
        start = time.time()

        _, _, fitness_curve = mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=pop, max_attempts=100,
                                                 max_iters=10000, curve=True)
        end = time.time() - start
        return str(pop), fitness_curve, end

    def run_single_hyper_mimic_keep(self, keep):
        problem, init_state = self.ProblemConstructor()
        start = time.time()
        _, _, fitness_curve = mlrose_hiive.mimic(problem, keep_pct=keep, pop_size=200, max_attempts=100,
                                                 max_iters=10000, curve=True)
        end = time.time() - start
        return str(keep), fitness_curve, end

    def run_hyperparameters(self):
        self.hyperparam_rhc()
        self.hyperparam_sa()
        self.hyperparam_ga()
        self.hyperparam_mimic()
        return
