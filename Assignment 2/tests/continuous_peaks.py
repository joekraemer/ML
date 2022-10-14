import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import time
from random import randint
import warnings

from util.graphing import plot_lc_iterations, plot_fitness_vs_complexity, plot_time_vs_complexity, plot_lc_evaluations, \
    plot_hyperparam_dict_generic, plot_helper

from tests.hyperparameter_tester import HyperTester

DATASET = 'continuous_peaks'


def construct_standard_cont_peaks(complexity=100):
    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    problem = mlrose_hiive.DiscreteOpt(length=complexity, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=complexity)
    return problem, init_state


def custom_fitness_callback(iteration=None, attempt=None, done=None, state=None, fitness=None, fitness_evaluations=None, curve=None, user_data=None):
        """state_fitness_callback: function taking five parameters, default: None
        If specified, this callback will be invoked once per iteration.
        Parameters are (iteration, max attempts reached?, current best state, current best fit, user callback data).
        Return true to continue iterating, or false to stop."""

        # I'm guessing that current_best_fit is referring to best_fitness
        look_back_depth = 1000
        conv_threshold = 0.01

        # I'm just going to do a simple convergence criteria based on the standard deviation
        if iteration > look_back_depth:
            std_dev = curve[iteration - look_back_depth:iteration, 0].std()
            if std_dev < conv_threshold:
                print('breakout')
                return False

        return True


def exp_problem_complexity():
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

    complexity_list = range(10, 125, 30)

    for c in complexity_list:
        problem, init_state = construct_standard_cont_peaks(c)

        start = time.time()
        _, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                                init_state=init_state, curve=True, state_fitness_callback=custom_fitness_callback, callback_user_info=[])
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

        fitness_dict['rhc'].append(best_fitness_rhc)
        fitness_dict['sa'].append(best_fitness_sa)
        fitness_dict['ga'].append(best_fitness_ga)
        fitness_dict['mimic'].append(best_fitness_mimic)

        time_dict['rhc'].append(rhc_time)
        time_dict['sa'].append(sa_time)
        time_dict['ga'].append(ga_time)
        time_dict['mimic'].append(mimic_time)

    plot_fitness_vs_complexity(fitness_dict, complexity_list, DATASET)
    plot_time_vs_complexity(time_dict, complexity_list, DATASET)

    return


def exp_iterations(problem_length=100):
    ## Plot change with respect to iterations
    problem, init_state = construct_standard_cont_peaks(problem_length)


    _, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=100000,
                                                             init_state=init_state, curve=True, state_fitness_callback=custom_fitness_callback, callback_user_info=[])
    print("Done with RHC iterations!")
    _, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                              max_attempts=1000, max_iters=100000,
                                                              init_state=init_state, curve=True, state_fitness_callback=custom_fitness_callback, callback_user_info=[])
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


def change_in_threshold():
    # TODO: Don't repeat yourself, de-plagerize
    ## Plot change with respect to different threshold values in the 4-Peaks problem

    problem_length = 40
    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.1)
    problem = mlrose_hiive.DiscreteOpt(length=problem_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)
    _, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                max_attempts=1000, max_iters=100000,
                                                                init_state=init_state, curve=True)
    _, _, fitness_curve_rhc_1 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                               init_state=init_state, curve=True)
    _, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=10000, curve=True)
    _, _, fitness_curve_mimic_1 = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000,
                                                     curve=True)
    print("Done with threshold 1!")

    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.3)
    problem = mlrose_hiive.DiscreteOpt(length=problem_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)
    _, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                max_attempts=1000, max_iters=100000,
                                                                init_state=init_state, curve=True)
    _, _, fitness_curve_rhc_2 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                               init_state=init_state, curve=True)
    _, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=10000, curve=True)
    _, _, fitness_curve_mimic_2 = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000,
                                                     curve=True)
    print("Done with threshold 2!")

    fitness = mlrose_hiive.ContinuousPeaks(t_pct=0.5)
    problem = mlrose_hiive.DiscreteOpt(length=problem_length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)
    _, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                                max_attempts=1000, max_iters=100000,
                                                                init_state=init_state, curve=True)
    _, _, fitness_curve_rhc_3 = mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=10000,
                                                               init_state=init_state, curve=True)
    _, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(problem, max_attempts=1000, max_iters=10000, curve=True)
    _, _, fitness_curve_mimic_3 = mlrose_hiive.mimic(problem, pop_size=500, max_attempts=100, max_iters=10000,
                                                     curve=True)
    print("Done with threshold 3!")

    plt.figure()
    plt.subplot(311)
    plt.plot(fitness_curve_sa_1[:, 0], label='SA, t = 0.1')
    plt.plot(fitness_curve_rhc_1[:, 0], label='RHC, t = 0.1')
    plt.plot(fitness_curve_ga_1[:, 0], label='GA, t = 0.1')
    plt.plot(fitness_curve_mimic_1[:, 0], label='MIMIC, t = 0.1')
    plt.title('Fitness Curve for Varying Thresholds (Continuous Peaks)')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(312)
    plt.plot(fitness_curve_sa_2[:, 0], label='SA, t = 0.3')
    plt.plot(fitness_curve_rhc_2[:, 0], label='RHC, t = 0.3')
    plt.plot(fitness_curve_ga_2[:, 0], label='GA, t = 0.3')
    plt.plot(fitness_curve_mimic_2[:, 0], label='MIMIC, t = 0.3')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.subplot(313)
    plt.plot(fitness_curve_sa_3[:, 0], label='SA, t = 0.5')
    plt.plot(fitness_curve_rhc_3[:, 0], label='RHC, t = 0.5')
    plt.plot(fitness_curve_ga_3[:, 0], label='GA, t = 0.5')
    plt.plot(fitness_curve_mimic_3[:, 0], label='MIMIC, t = 0.5')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig('continuous_peaks_threshold.png')


def hyperparam_rhc(problem, init_state):
    fitness_curve_dict = {}

    num_restarts = [0, 4, 8, 12, 16]

    for r in num_restarts:
        _, _, fitness_curve_dict[str(r)] = mlrose_hiive.random_hill_climb(problem, restarts=r, max_attempts=1000,
                                                                          max_iters=10000,
                                                                          init_state=init_state, curve=True)

    print("Completed RHC hyper-parameter testing!")

    # plotting
    plot_hyperparam_dict_generic(fitness_curve_dict, label='# restarts = ')
    plot_helper('', DATASET + '_hyperparam_rhc_restarts', DATASET)
    return


def hyperparam_sa(problem, init_state):
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
    plot_helper('', DATASET + '_hyperparam_sa_decay', DATASET)
    return


def hyperparam_ga(problem, init_state):
    fitness_curve_dict = {}
    pop_size = [50, 100, 200, 350, 500]

    for pop in pop_size:
        _, _, fitness_curve_dict[str(pop)] = mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=pop,
                                                                      max_attempts=1000,
                                                                      max_iters=10000, curve=True)

    print("Completed GA pop size testing")

    plot_hyperparam_dict_generic(fitness_curve_dict, label='Population = ')
    plot_helper('', DATASET + '_hyperparam_ga_pop_size', DATASET)

    #################

    fitness_curve_dict = {}
    mutation_prob = [0.01, 0.05, 0.1, 0.2]

    for mut in mutation_prob:
        _, _, fitness_curve_dict[str(mut)] = mlrose_hiive.genetic_alg(problem, mutation_prob=mut, pop_size=100,
                                                                      max_attempts=1000,
                                                                      max_iters=10000, curve=True)

    print("Completed GA mutation prob testing")
    plot_hyperparam_dict_generic(fitness_curve_dict, label='Mutation Prob = ')
    plot_helper('', DATASET + '_hyperparam_ga_mutation_prob', DATASET)

    return


def hyperparam_mimic(problem, init_state):
    fitness_curve_dict = {}
    pop_size = [100, 200, 400, 500]

    for pop in pop_size:
        _, _, fitness_curve_dict[str(pop)] = mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=pop, max_attempts=100,
                                                                max_iters=10000, curve=True)

    print("Completed MIMIC pop size testing")

    plot_hyperparam_dict_generic(fitness_curve_dict, label='population = ')
    plot_helper('', DATASET + '_hyperparam_mimic_pop_size', DATASET)

    #################

    fitness_curve_dict = {}
    keep_percent = [0.05, 0.1, 0.2, 0.5]

    for kp in keep_percent:
        _, _, fitness_curve_dict[str(kp)] = mlrose_hiive.mimic(problem, keep_pct=kp, pop_size=200, max_attempts=100,
                                                               max_iters=10000, curve=True)

    print("Completed MIMIC keep % testing")
    plot_hyperparam_dict_generic(fitness_curve_dict, label='keep % = ')
    plot_helper('', DATASET + '_hyperparam_mimic_keep_percent', DATASET)
    return


def run_hyperparameters():
    problem, init_state = construct_standard_cont_peaks(100)

    hyperparam_rhc(problem, init_state)
    hyperparam_sa(problem, init_state)
    hyperparam_ga(problem, init_state)
    hyperparam_mimic(problem, init_state)



def run_continuous_peaks():
    np.random.seed(69)

    exp_problem_complexity()
    exp_iterations()

    run_hyperparameters()

    hyperTester = HyperTester(construct_standard_cont_peaks, 'continuous_peaks')
    hyperTester.run_hyperparameters()

if __name__ == '__main__':
    run_continuous_peaks()
