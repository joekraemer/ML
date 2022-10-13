import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import time
from random import randint
import warnings

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity


def exp_problem_complexity():

    fitness_dict = {
        "sa": [],
        "rhc": [],
        "ga": [],
        "mimic": [],
    }

    time_dict = {
        "sa": [],
        "rhc": [],
        "ga": [],
        "mimic": [],
    }

    complexity_list = range(5, 125, 40)

    for c in complexity_list:
        fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
        problem = mlrose_hiive.DiscreteOpt(length=c, fitness_fn=fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=c)

        start = time.time()
        _, best_fitness_sa, _ = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                 max_attempts=1000, max_iters=100000,
                                                                 init_state=init_state, curve=True)
        end = time.time()
        sa_time = end - start
        print("SA:", sa_time, c)

        start = time.time()
        _, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                                init_state=init_state, curve=True)
        end = time.time()
        rhc_time = end - start
        print("RHC:", rhc_time, c)

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

        fitness_dict['sa'].append(best_fitness_sa)
        fitness_dict['rhc'].append(best_fitness_rhc)
        fitness_dict['ga'].append(best_fitness_ga)
        fitness_dict['mimic'].append(best_fitness_mimic)

        time_dict['sa'].append(sa_time)
        time_dict['rhc'].append(rhc_time)
        time_dict['ga'].append(ga_time)
        time_dict['mimic'].append(mimic_time)

    plot_fitness_vs_complexity(fitness_dict, complexity_list, 'continuous_peaks')
    plot_time_vs_complexity(time_dict, complexity_list, 'continuous_peaks')

    return


def exp_iterations(problem_length = 100):
    iterations_dict = {
        "sa": [],
        "rhc": [],
        "ga": [],
        "mimic": [],
    }

    ## Plot change with respect to iterations
    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    problem = mlrose_hiive.DiscreteOpt(length=problem_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)
    _, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                              max_attempts=1000, max_iters=100000,
                                                              init_state=init_state, curve=True)
    print("Done with SA iterations!")
    _, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                             init_state=init_state, curve=True)
    print("Done with RHC iterations!")
    _, _, fitness_curve_ga = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=10000, curve=True)
    print("Done with GA iterations!")
    _, _, fitness_curve_mimic = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000, curve=True)
    print("Done with MIMIC iterations!")


    iterations_dict = {
        "sa": fitness_curve_sa[:, 0],
        "rhc": fitness_curve_rhc[:, 0],
        "ga": fitness_curve_ga[:, 0],
        "mimic": fitness_curve_mimic[:, 0],
    }

    plot_lc_iterations(iterations_dict, 'continuous_peaks')


def change_in_threshold():



def run_continuous_peaks():
    np.random.seed(69)

    exp_problem_complexity()
    exp_iterations()


if __name__ == '__main__':
    run_continuous_peaks()
