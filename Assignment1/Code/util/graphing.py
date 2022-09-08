import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_learning_curve(train_scores, test_scores, train_sizes, name):
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
    ax.set_ylabel("Score")

    plot_helper('Learning Curve for ' + name, 'lc_' + name, show=False)

    return


def plot_hyperparam_validation_curve(train_scores, test_scores, train_sizes, name):
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
    ax.set_ylabel("Score")

    plot_helper('Learning Curve for ' + name, 'lc_' + name, show=False)

    return


def plot_scalability(fit_times, train_sizes, name):
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
    ax.set_ylabel("Time to Fit Model (ADD UNITS)")

    plot_helper('Scalability of ' + name, 'scalability_' + name, show=False)
    return


def plot_helper(title, filename, show=True):
    plt.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.11)
    plt.title(title, fontsize=12)
    plt.grid()
    if show:
        plt.show()

    root_dir = Path(".")
    path = root_dir / 'Graphs' / (filename + ".png")

    plt.savefig(path)
    plt.clf()
    return
