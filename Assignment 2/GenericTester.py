from abc import abstractmethod
import numpy as np
import mlrose_hiive
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import math

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper, plot_lc_fitness_vs_evals
from util.logging_tables import log_evals_table
from tests.hyperparameter_tester import HyperTester


def calc_max_attempts(n, sigmas=2):
    expectation = n * (math.log(math.e) * n + 0.577216) + 0.5

    sigma = math.sqrt()
    sigma = (math.pi ** 2) / 6 * (n ** 2)


def calc_sigma(p):
    return math.sqrt((1 - p) / (p ** 2))


class GenericTester(object):
    def __init__(self, name, complexity_list=range(25, 250, 40), aws=False, debug=False, hyper_config=None):
        num_cores = multiprocessing.cpu_count()
        self.Name = name
        if aws:
            self.N_Jobs = num_cores
        else:
            self.N_Jobs = num_cores - 1
        self.ComplexityList = complexity_list
        self.Seed = 123456
        self.Debug = debug
        self.HyperConfig = hyper_config

        if self.Debug:
            self.NumberOfRuns = 2
        else:
            self.NumberOfRuns = 10

        return

    @abstractmethod
    def problem_constructor(self, complexity=20, seed=123456):
        problem = None
        init_state = None
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):

        # max_attempts = calc_max_attempts(problem.complexity)

        return mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=20000, restarts=10,
                                              init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                max_attempts=500, max_iters=20000,
                                                init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        return mlrose_hiive.genetic_alg(problem, max_attempts=500, max_iters=20000, curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        return mlrose_hiive.mimic(problem, pop_size=150, max_attempts=500, max_iters=20000,
                                  curve=curve)

    def run(self):
        start = time.time()
        print("Running " + self.Name)
        self.run_experiment_complexity()
        print(self.Name + " Complexity Completed")
        self.run_experiment_iterations()
        print(self.Name + " Iterations Completed")
        self.run_hyperparameters()
        print(self.Name + " Hyperparameters Completed")
        self.run_extra()
        print(self.Name + " Extra Tests Completed")
        total_time = time.time() - start
        print(self.Name + "Run Time:", total_time)

        return

    def run_experiment_complexity(self):
        inputs = tqdm(self.ComplexityList)

        fitness_dict = {
            "rhc": [],
            "sa": [],
            "ga": [],
            "mimic": [],
        }

        time_dict = {
            "rhc": [],
            "sa": [],
            "ga": [],
            "mimic": [],
        }

        for n in range(0, self.NumberOfRuns):
            processed_list_all = Parallel(n_jobs=self.N_Jobs)(
                delayed(self._run_single_complexity)(i, n) for i in inputs)
            print("all complete")

            # peel off the evals first
            # for run in processed_list_all:
            #    evals_rhc = run[:, 8]
            #    diff = np.diff(evals_rhc)
            #    avg = diff.mean()
            #    std = diff.std()

            as_array = np.array(processed_list_all)

            fitness_dict["rhc"].append(as_array[:, 4])
            fitness_dict["sa"].append(as_array[:, 5])
            fitness_dict["ga"].append(as_array[:, 6])
            fitness_dict["mimic"].append(as_array[:, 7])

            time_dict["rhc"].append(as_array[:, 0])
            time_dict["sa"].append(as_array[:, 1])
            time_dict["ga"].append(as_array[:, 2])
            time_dict["mimic"].append(as_array[:, 3])

        # plot_evaluations_vs_complexity(fitness_dict, self.ComplexityList, self.Name)
        plot_fitness_vs_complexity(fitness_dict, self.ComplexityList, self.Name)
        plot_time_vs_complexity(time_dict, self.ComplexityList, self.Name)

        return

    def _run_single_complexity(self, c, seed_offset):
        seed_offset = seed_offset * 1234
        problem, init_state = self.problem_constructor(c, self.Seed + seed_offset)

        start = time.time()
        _, best_fitness_rhc, fc_rhc = self.run_best_rhc(problem, init_state, curve=True)
        rhc_time = time.time() - start
        print("RHC:", rhc_time, c)

        start = time.time()
        _, best_fitness_sa, fc_sa = self.run_best_sa(problem, init_state, curve=True)
        sa_time = time.time() - start
        print("SA:", sa_time, c)

        start = time.time()
        _, best_fitness_ga, fc_ga = self.run_best_ga(problem, init_state, curve=True)
        ga_time = time.time() - start
        print("GA:", ga_time, c)

        start = time.time()
        _, best_fitness_mimic, fc_mimic = self.run_best_mimic(problem, init_state, curve=True)
        mimic_time = time.time() - start
        print("MIMIC:", mimic_time, c)

        # total_evals = fc[-1] - fc[0]
        # total_iters =

        return [rhc_time, sa_time, ga_time, mimic_time, best_fitness_rhc, best_fitness_sa, best_fitness_ga,
                best_fitness_mimic, fc_rhc[:, 1], fc_sa[:, 1], fc_ga[:, 1], fc_mimic[:, 1]]

    def run_experiment_iterations(self):
        inputs = tqdm(range(0, self.NumberOfRuns))

        ## Plot change with respect to iterations
        fitness_dict = {
            "rhc": [],
            "sa": [],
            "ga": [],
            "mimic": [],
        }

        evaluations_dict = {
            "rhc": [],
            "sa": [],
            "ga": [],
            "mimic": [],
        }

        times_dict = {
            "rhc": [],
            "sa": [],
            "ga": [],
            "mimic": [],
        }

        processed_list_all = Parallel(n_jobs=self.N_Jobs)(
            delayed(self._run_single_iterations)(i) for i in inputs)
        print("All iterations experiments complete")

        for run_res in processed_list_all:
            fitness_dict["rhc"].append(run_res[0])
            fitness_dict["sa"].append(run_res[1])
            fitness_dict["ga"].append(run_res[2])
            fitness_dict["mimic"].append(run_res[3])

            evaluations_dict["rhc"].append(run_res[4])
            evaluations_dict["sa"].append(run_res[5])
            evaluations_dict["ga"].append(run_res[6])
            evaluations_dict["mimic"].append(run_res[7])

            times_dict["rhc"].append(run_res[8])
            times_dict["sa"].append(run_res[9])
            times_dict["ga"].append(run_res[10])
            times_dict["mimic"].append(run_res[11])

        log_evals_table(evaluations_dict, times_dict, name=self.Name)
        plot_lc_iterations(fitness_dict, self.Name)
        plot_lc_evaluations(evaluations_dict, self.Name)
        try:
            plot_lc_fitness_vs_evals(fitness_dict, evaluations_dict, self.Name)
        except:
            print("Plotting Evals didn't work")
        return

    def _run_single_iterations(self, seed_offset):
        seed_offset = seed_offset * 1234
        problem, init_state = self.problem_constructor(seed=self.Seed + seed_offset * 1000)
        start = time.time()
        _, _, fc_rhc = self.run_best_rhc(problem, init_state, curve=True)
        rhc_time = time.time() - start
        print("Done with RHC iterations!")

        problem, init_state = self.problem_constructor(seed=self.Seed + seed_offset)
        start = time.time()
        _, _, fc_sa = self.run_best_sa(problem, init_state, curve=True)
        sa_time = time.time() - start
        print("Done with SA iterations!")

        problem, init_state = self.problem_constructor(seed=self.Seed + seed_offset)
        start = time.time()
        _, _, fc_ga = self.run_best_ga(problem, init_state, curve=True)
        ga_time = time.time() - start
        print("Done with GA iterations!")

        problem, init_state = self.problem_constructor(seed=self.Seed + seed_offset)
        start = time.time()
        _, _, fc_mimic = self.run_best_mimic(problem, init_state, curve=True)
        mimic_time = time.time() - start
        print("Done with MIMIC iterations!")

        # (column 0 is fitness per iteration and column 1 is total evaluations per iteration)
        return fc_rhc[:, 0], fc_sa[:, 0], fc_ga[:, 0], fc_mimic[:, 0], fc_rhc[:, 1], fc_sa[:, 1], \
               fc_ga[:, 1], fc_mimic[:, 1], rhc_time, sa_time, ga_time, mimic_time

    def run_hyperparameters(self):
        hyperTester = HyperTester(self.problem_constructor, self.Name, config=self.HyperConfig)
        hyperTester.run_hyperparameters()
        return

    @abstractmethod
    def run_extra(self):
        return
