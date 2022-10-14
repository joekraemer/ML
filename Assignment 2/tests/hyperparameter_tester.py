import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper


N_JOBS = 7


class HyperTester(object):
    def __init__(self, problem_constructor, dataset, n_runs=3):
        self.ProblemConstructor = problem_constructor
        self.Dataset = dataset
        self.N_Runs = n_runs
        return

    def generic_hyperparam_multi(self, problem, init_state, function, params, legend_base_label, filename_base, dataset):
        fitness_curve_dict = {}

        for p in params:
            fitness_curve_dict[str(p)] = []

        inputs = tqdm(params)

        for _ in range(0, self.N_Runs):

            processed_list_all = Parallel(n_jobs=N_JOBS)(delayed(function)(i, problem, init_state) for i in inputs)

            for res in processed_list_all:
                fitness_curve_dict[res[0]].append(res[1][:, 0]) # TODO split this to just grab fitness

        # plotting
        plot_hyperparam_dict_generic(fitness_curve_dict, label=legend_base_label)
        plot_helper('', dataset + '_hyperparam' + filename_base, dataset)

    def hyperparam_rhc(self, problem, init_state):
        num_restarts = [0, 4, 8, 12, 16]
        self.generic_hyperparam_multi(problem, init_state, function=self.run_single_hyper_rhc, params=num_restarts, legend_base_label='# restarts = ', filename_base='_rhc_restarts', dataset=self.Dataset )

        print("Completed RHC hyper-parameter testing!")
        return


    def run_single_hyper_rhc(self,r, problem, init_state):
        _, _, fitness_curve = mlrose_hiive.random_hill_climb(problem, restarts=r, max_attempts=1000,
                                                                          max_iters=10000,
                                                                          init_state=init_state, curve=True)
        return str(r), fitness_curve


    def hyperparam_sa(self,problem, init_state):
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
        plot_helper('', self.Dataset  + '_hyperparam_sa_decay', self.Dataset )
        return


    def hyperparam_ga(self,problem, init_state):
        pop_size = [50, 100, 200, 350]
        self.generic_hyperparam_multi(problem, init_state, function=self.run_single_hyper_ga_pop, params=pop_size, legend_base_label='Population = ',
                                 filename_base='_ga_pop_size', dataset=self.Dataset )
        print("Completed GA population size hyper-parameter")

        mutation_prob = [0.02, 0.05, 0.1, 0.2]
        self.generic_hyperparam_multi(problem, init_state, function=self.run_single_hyper_ga_mut, params=mutation_prob, legend_base_label='Mutation Prob = ',
                                 filename_base='_ga_mutation_prob', dataset=self.Dataset )
        print("Completed GA mutation prob testing")
        return


    def run_single_hyper_ga_pop(self,pop, problem, init_state):
        _, _, fitness_curve = mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=pop,
                                                                      max_attempts=1000,
                                                                      max_iters=10000, curve=True)
        return str(pop), fitness_curve


    def run_single_hyper_ga_mut(self,mut, problem, init_state):
        _, _, fitness_curve = mlrose_hiive.genetic_alg(problem, mutation_prob=mut, pop_size=100,
                                                                          max_attempts=1000,
                                                                          max_iters=10000, curve=True)
        return str(mut), fitness_curve


    def hyperparam_mimic(self,problem, init_state):
        pop_size = [100, 200, 300, 400]
        self.generic_hyperparam_multi(problem, init_state, function=self.run_single_hyper_mimic_pop, params=pop_size,
                                 legend_base_label='Population = ',
                                 filename_base='_mimic_pop_size', dataset=self.Dataset )
        print("Completed MIMIC population size hyper-parameter")

        keep_percent = [0.05, 0.1, 0.2, 0.3]
        self.generic_hyperparam_multi(problem, init_state, function=self.run_single_hyper_mimic_keep, params=keep_percent,
                                 legend_base_label='Keep % = ',
                                 filename_base='_mimic_keep_percent', dataset=self.Dataset )
        print("Completed MIMIC keep % testing")


    def run_single_hyper_mimic_pop(self,pop, problem, init_state):
        _, _, fitness_curve = mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=pop, max_attempts=100,
                                                                max_iters=10000, curve=True)
        return str(pop), fitness_curve


    def run_single_hyper_mimic_keep(self,keep, problem, init_state):
        _, _, fitness_curve = mlrose_hiive.mimic(problem, keep_pct=keep, pop_size=200, max_attempts=100,
                                                                max_iters=10000, curve=True)
        return str(keep), fitness_curve


    def run_hyperparameters(self):
        problem, init_state = self.ProblemConstructor()

        self.hyperparam_rhc(problem, init_state)
        self.hyperparam_sa(problem, init_state)
        self.hyperparam_ga(problem, init_state)
        self.hyperparam_mimic(problem, init_state)
        return
