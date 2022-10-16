import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd


def _draw_variance_curve(ax, df, x=None, label=''):
    data_mean = np.mean(df, axis=0)
    data_std = np.std(df, axis=0)
    if x is None:
        x = range(0, len(data_mean))

    # Plot learning curve
    ax.fill_between(
        x,
        data_mean - data_std,
        data_mean + data_std,
        alpha=0.1
    )

    ax.plot(x, data_mean, label=label)
    return


def plot_algo_dict_generic_with_variance(data_dict, x=None, height=3.5, width=5):
    fig, ax = plt.subplots()

    rhc = pd.DataFrame(data_dict['rhc'])
    sa = pd.DataFrame(data_dict['sa'])
    ga = pd.DataFrame(data_dict['ga'])
    mimic = pd.DataFrame(data_dict['mimic'])

    _draw_variance_curve(ax, rhc, x, label='Randomized Hill Climb')
    _draw_variance_curve(ax, sa, x, label='Simulated Annealing')
    _draw_variance_curve(ax, ga, x, label='Genetic Algorithm')
    _draw_variance_curve(ax, mimic, x, label='MIMIC')

    return fig, ax


def plot_algo_dict_generic(dict, x=None):
    # plots generic data from a set dictionary config

    rhc = np.array(dict['rhc'])
    sa = np.array(dict['sa'])
    ga = np.array(dict['ga'])
    mimic = np.array(dict['mimic'])

    if x is not None:
        plt.plot(x, rhc, label='Randomized Hill Climb')
        plt.plot(x, sa, label='Simulated Annealing')
        plt.plot(x, ga, label='Genetic Algorithm')
        plt.plot(x, mimic, label='MIMIC')
    else:
        plt.plot(rhc, label='Randomized Hill Climb')
        plt.plot(sa, label='Simulated Annealing')
        plt.plot(ga, label='Genetic Algorithm')
        plt.plot(mimic, label='MIMIC')
    return


def plot_hyperparam_dict_generic(hyper_dict, label, height=3.5, width=5):
    fig, ax = plt.subplots()

    for key, item in hyper_dict.items():
        df = pd.DataFrame(item)
        label_temp = label + key
        _draw_variance_curve(ax, df, label=label_temp)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Fitness')
    ax.legend(loc="best")
    return


def plot_loss_curves(loss_dict, dataset, height=3.5, width=5):
    fig, ax = plt.subplots()

    rhc = pd.DataFrame(loss_dict['rhc'])
    sa = pd.DataFrame(loss_dict['sa'])
    ga = pd.DataFrame(loss_dict['ga'])
    gd = pd.DataFrame(loss_dict['gd'])

    _draw_variance_curve(ax, rhc, label='Randomized Hill Climb')
    _draw_variance_curve(ax, sa, label='Simulated Annealing')
    _draw_variance_curve(ax, ga, label='Genetic Algorithm')
    _draw_variance_curve(ax, gd, label='Gradient Descent')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plot_helper('', dataset + '_loss_vs_iterations', 'nn_' + dataset)
    return


def plot_lc_evaluations(evaluations_dict, dataset):
    plt.figure()
    plot_algo_dict_generic_with_variance(evaluations_dict)

    plt.xlabel('Iterations')
    plt.ylabel('# of Function Evaluations')
    plt.legend()
    plot_helper('', dataset + '_evaluations_vs_iterations', dataset)
    return


def plot_lc_fitness_vs_evals(fitness_dict, evals_dict, dataset):
    plt.figure()

    rhc = fitness_dict['rhc'][0]
    rhc_x = evals_dict['rhc'][0]
    rhc_x = rhc_x - rhc_x[0]
    sa = fitness_dict['sa'][0]
    sa_x = evals_dict['sa'][0]
    sa_x = sa_x - sa_x[0]

    ga = fitness_dict['ga'][0]
    ga_x = evals_dict['ga'][0]
    mimic = fitness_dict['mimic'][0]
    mimic_x = evals_dict['mimic'][0]

    plt.plot(rhc_x, rhc, label='Randomized Hill Climb')
    plt.plot(sa_x, sa, label='Simulated Annealing')
    plt.plot(ga_x, ga, label='Genetic Algorithm')
    plt.plot(mimic_x, mimic, label='MIMIC')

    plt.xlabel('# of Function Evaluations')
    plt.ylabel('Fitness')
    plt.legend()
    plot_helper('', dataset + '_fitness_vs_evaluations', dataset)


def plot_lc_iterations(iterations_dict, dataset):
    plt.figure()
    plot_algo_dict_generic_with_variance(iterations_dict)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plot_helper('', dataset + '_fitness_vs_iterations', dataset)
    return


def plot_generic_multi_algo_dict(data_dict, x, dataset, xlabel, ylabel, filename, height=3.5, width=5):
    # catch when we are trying to plot variance
    if type(data_dict['rhc']) == list:
        fig, ax = plot_algo_dict_generic_with_variance(data_dict, x)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    else:
        plt.figure()
        plot_algo_dict_generic(data_dict, x)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

    plot_helper('', dataset + filename, dataset)
    return


def plot_fitness_vs_complexity(complexity_dict, x, dataset):
    plot_generic_multi_algo_dict(complexity_dict, x, dataset,
                                 xlabel='Problem Size',
                                 ylabel='Fitness',
                                 filename='_fitness_vs_complexity_')
    return


def plot_time_vs_complexity(time_dict, x, dataset):
    plot_generic_multi_algo_dict(time_dict, x, dataset,
                                 xlabel='Problem Size',
                                 ylabel='Computation Time (s)',
                                 filename='_time_vs_comlexity')
    return


def plot_learning_curve(train_scores, test_scores, train_sizes, name, folder, height=3.5, width=5):
    fig, ax = plt.subplots()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    ax.legend(loc="best")

    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 Weighted Score")

    plot_helper('', 'learning_curve_' + name, folder=folder, show=False)
    return


def plot_scalability(fit_times, train_sizes, name, folder):
    # Plot Scalability
    fig, ax = plt.subplots()

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    ax.grid()
    ax.plot(train_sizes, fit_times_mean, "o-")
    ax.fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Time to Fit Model (s)")

    plot_helper('', 'scalability_' + name, folder=folder, show=False)
    return


def bar_chart(labels, data_1, yaxis_label, d1_std=None,
                      title='', file_name='', folder=None):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    x = np.arange(len(labels)).astype(float)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    rects1 = ax.bar(x, data_1, width, yerr=d1_std, color='g')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(yaxis_label)
    ax.legend()

    plot_helper(title, file_name, folder=folder, show=False)
    return


def grouped_bar_chart(labels, data_1, data_2, label_1, label_2, yaxis_label, d1_std=None, d2_std=None,
                      title='', file_name='', folder=None, logy=False):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    x = np.arange(len(labels)).astype(float)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    rects1 = ax.bar(x - width/2, data_1, width, yerr=d1_std, label=label_1, color='r')
    rects2 = ax.bar(x + width/2, data_2, width, yerr=d2_std, label=label_2, color='g')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(yaxis_label)
    ax.legend()

    if logy:
        plt.semilogy()

    plot_helper(title, file_name, folder=folder, show=False)
    return


def plot_single_curve(data, name, title, ylabel, xlabel, folder):
    fig, ax = plt.subplots()

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    x = range(0, len(data_mean))

    # Plot learning curve
    ax.fill_between(
        x,
        data_mean - data_std,
        data_mean + data_std,
        alpha=0.1,
        color="r",
    )

    ax.plot(
        x, data_mean, color="r")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plot_helper(title, name, folder=folder, show=False)

    return


def plot_helper(title, filename, folder, show=False):
    plt.tight_layout()
    if title == '':
        plt.subplots_adjust(left=0.145, right=0.985, top=0.985, bottom=0.11)
    else:
        plt.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.11)
        plt.title(title, fontsize=12)
    plt.grid()
    if show:
        plt.show()

    root_dir = Path(".")
    path = root_dir / 'Graphs' / folder / (filename + ".png")

    plt.savefig(path)
    plt.clf()
    return
