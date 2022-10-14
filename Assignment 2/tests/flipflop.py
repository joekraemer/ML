import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import time
import datetime
import multiprocessing

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper
from tests.hyperparameter_tester import HyperTester

DATASET = 'flipflop'
N_JOBS = 7


def construct_standard_flipflop(complexity=120):
    fitness = mlrose_hiive.FlipFlop()
    problem = mlrose_hiive.DiscreteOpt(length=complexity, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=complexity)
    return problem, init_state


def exp_problem_complexity_with_parallel():
    complexity_list = range(25, 250, 40)

    inputs = tqdm(complexity_list)

    processed_list_all = Parallel(n_jobs=N_JOBS)(delayed(run_single_complexity)(i) for i in inputs)
    print("all complete")

    as_array = np.array(processed_list_all)

    fitness_dict = {
        "rhc": as_array[:, 4],
        "sa": as_array[:, 5],
        "ga": as_array[:, 6],
        "mimic": as_array[:, 7],
    }

    time_dict = {
        "rhc": as_array[:, 0],
        "sa": as_array[:, 1],
        "ga": as_array[:, 2],
        "mimic": as_array[:, 3],
    }

    plot_fitness_vs_complexity(fitness_dict, complexity_list, DATASET)
    plot_time_vs_complexity(time_dict, complexity_list, DATASET)

    return


def run_single_complexity(c):
    problem, init_state = construct_standard_flipflop(c)

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

    return [rhc_time, sa_time, ga_time, mimic_time, best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_mimic]


def exp_iterations():
    ## Plot change with respect to iterations
    problem, init_state = construct_standard_flipflop()

    _, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                             init_state=init_state, curve=True)
    print("Done with RHC iterations!")
    _, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                              max_attempts=1000, max_iters=100000,
                                                              init_state=init_state, curve=True)
    print("Done with SA iterations!")

    _, _, fitness_curve_ga = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=1000, curve=True)
    print("Done with GA iterations!")
    _, _, fitness_curve_mimic = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000, curve=True)
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

    plot_lc_iterations(iterations_dict, DATASET)
    plot_lc_evaluations(evaluations_dict, DATASET)


def run_flipflop():
    np.random.seed(12435)

    exp_problem_complexity_with_parallel()
    exp_iterations()

    hyperTester = HyperTester(construct_standard_flipflop, DATASET)
    hyperTester.run_hyperparameters()


if __name__ == '__main__':
    run_flipflop()
