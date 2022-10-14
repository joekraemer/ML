from abc import abstractmethod
import numpy as np
import mlrose_hiive
import time
from joblib import Parallel, delayed
from tqdm import tqdm

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper
from tests.hyperparameter_tester import HyperTester


class GenericTester(object):
    def __init__(self, name, n_jobs=7, complexity_list=range(25, 250, 40)):
        self.Name = name
        self.N_Jobs = n_jobs
        self.ComplexityList = complexity_list
        return

    @abstractmethod
    def problem_constructor(self, complexity=20):
        problem = None
        init_state = None
        return problem, init_state

    def run(self):
        self.run_single_exp_complexity()
        self.run_single_exp_iterations()
        self.run_hyperparameters()

    def run_single_exp_complexity(self, number_of_runs=3):
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

        for _ in range(0, number_of_runs):
            processed_list_all = Parallel(n_jobs=self.N_Jobs)(delayed(self._run_single_complexity)(i) for i in inputs)
            print("all complete")

            as_array = np.array(processed_list_all)

            fitness_dict["rhc"].append(as_array[:, 4])
            fitness_dict["sa"].append(as_array[:, 5])
            fitness_dict["ga"].append(as_array[:, 6])
            fitness_dict["mimic"].append(as_array[:, 7])

            time_dict["rhc"].append(as_array[:, 0])
            time_dict["sa"].append(as_array[:, 1])
            time_dict["ga"].append(as_array[:, 2])
            time_dict["mimic"].append(as_array[:, 3])

        plot_fitness_vs_complexity(fitness_dict, self.ComplexityList, self.Name)
        plot_time_vs_complexity(time_dict, self.ComplexityList, self.Name)

        return

    def _run_single_complexity(self, c):
        problem, init_state = self.problem_constructor(c)

        start = time.time()
        _, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                                init_state=init_state, curve=True)
        end = time.time()
        rhc_time = end - start
        print("RHC:", rhc_time, c)

        start = time.time()
        _, best_fitness_sa, _ = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                 max_attempts=1000, max_iters=100000,
                                                                 init_state=init_state, curve=True)
        end = time.time()
        sa_time = end - start
        print("SA:", sa_time, c)

        start = time.time()
        _, best_fitness_ga, _ = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=10000, curve=True)
        end = time.time()
        ga_time = end - start
        print("GA:", ga_time, c)

        start = time.time()
        _, best_fitness_mimic, _ = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000,
                                                      curve=True)
        end = time.time()
        mimic_time = end - start
        print("MIMIC:", mimic_time, c)

        return [rhc_time, sa_time, ga_time, mimic_time, best_fitness_rhc, best_fitness_sa, best_fitness_ga,
                best_fitness_mimic]

    def run_single_exp_iterations(self):
        ## Plot change with respect to iterations
        problem, init_state = self.problem_constructor()

        _, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                                 init_state=init_state, curve=True)
        print("Done with RHC iterations!")
        _, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                  max_attempts=1000, max_iters=100000,
                                                                  init_state=init_state, curve=True)
        print("Done with SA iterations!")

        _, _, fitness_curve_ga = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=1000, curve=True)
        print("Done with GA iterations!")
        _, _, fitness_curve_mimic = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000,
                                                       curve=True)
        print("Done with MIMIC iterations!")

        # (column 0 is fitness per iteration and column 1 is total evaluations per iteration)
        iterations_dict = {
            "rhc": fitness_curve_rhc[:, 0],
            "sa": fitness_curve_sa[:, 0],
            "ga": fitness_curve_ga[:, 0],
            "mimic": fitness_curve_mimic[:, 0],
        }

        evaluations_dict = {
            "rhc": fitness_curve_rhc[:, 1],
            "sa": fitness_curve_sa[:, 1],
            "ga": fitness_curve_ga[:, 1],
            "mimic": fitness_curve_mimic[:, 1],
        }

        plot_lc_iterations(iterations_dict, self.Name)
        plot_lc_evaluations(evaluations_dict, self.Name)

    def run_hyperparameters(self):
        hyperTester = HyperTester(self.problem_constructor(), self.Name)
        hyperTester.run_hyperparameters()
        return

