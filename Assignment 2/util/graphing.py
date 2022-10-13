import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns


def plot_algo_dict_generic(dict, x=None):
    # plots generic data from a set dictionary config

    sa = np.array(dict['sa'])
    rhc = np.array(dict['rhc'])
    ga = np.array(dict['ga'])
    mimic = np.array(dict['mimic'])

    if x is not None:
        plt.plot(x, sa, label='Simulated Annealing')
        plt.plot(x, rhc, label='Randomized Hill Climb')
        plt.plot(x, ga, label='Genetic Algorithm')
        plt.plot(x, mimic, label='MIMIC')
    else:
        plt.plot(sa, label='Simulated Annealing')
        plt.plot(rhc, label='Randomized Hill Climb')
        plt.plot(ga, label='Genetic Algorithm')
        plt.plot(mimic, label='MIMIC')

    return


def plot_lc_iterations(iterations_dict, dataset):
    plt.figure()
    plot_algo_dict_generic(iterations_dict)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plot_helper('', dataset + 'iterations_vs_fitness_', dataset)
    return


def plot_fitness_vs_complexity(complexity_dict, x, dataset):
    plt.figure()
    plot_algo_dict_generic(complexity_dict, x)

    plt.xlabel('Problem Size')
    plt.ylabel('Fitness')
    plt.legend()

    plot_helper('', dataset + 'fitness_vs_complexity_', dataset)
    return


def plot_time_vs_complexity(time_dict, x, dataset):
    plt.figure()
    plot_algo_dict_generic(time_dict, x)

    plt.xlabel('Problem Size')
    plt.ylabel('Computation Time (s)')
    plt.legend()

    plot_helper('', dataset + 'time_vs_comlexity_', dataset)
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
