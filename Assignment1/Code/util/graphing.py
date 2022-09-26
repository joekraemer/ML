import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns


def plot_correlation_matrix(df, name, hue=None, folder='Data'):
    matrix = df.corr().round(2)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
    plot_helper('Correlation Matix of ' + name, 'corr_matrix_' + name, folder=folder, show=False)
    sns.pairplot(df, hue=hue)
    plot_helper('Pair Plot of ' + name, 'pair_plot_' + name, folder=folder, show=False)


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


def plot_learning_curve(train_scores, test_scores, train_sizes, name, folder, height=3, width=5):
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

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

    plot_helper('', 'lc_' + name, folder=folder, show=False)

    return


def plot_hyperparam_validation_curve(train_scores, test_scores, x, name, label, folder, x_tick_labels=None, enable_dots=True, height=3, width=5):
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.fill_between(
        x,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        x,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )

    if enable_dots:
        ax.plot(
            x, train_scores_mean, "o-", color="r", label="Training score"
        )
        ax.plot(
            x, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
    else:
        ax.plot(
            x, train_scores_mean, color="r", label="Training score"
        )
        ax.plot(
            x, test_scores_mean, color="g", label="Cross-validation score"
        )

    ax.legend(loc="best")

    if x_tick_labels != None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel(label)
    ax.set_ylabel("Score")

    plot_helper('', 'hyper_tune_' + name + '_'+ label, folder=folder, show=False)
    plt.clf()
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


def plot_scalability(fit_times, train_sizes, name, folder, height=3, width=5):
    # Plot Scalability
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

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
    ax.set_ylabel("Time to Fit Model (ms)")

    plot_helper('', 'scalability_' + name, folder=folder, show=False)
    return


def plot_hyperparam_validation_bar_chart(train_scores, test_scores, x_labels, name, label, folder, height=3, width=5):
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    x = np.arange(len(x_labels))  # the label locations
    width = 0.35  # the width of the bars

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    rects1 = ax.bar(x - width/2, train_scores_mean, width, label='Training score', color="r")
    rects2 = ax.bar(x + width/2, test_scores_mean, width, label='Cross-validation score', color="g")

    ax.legend(loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.set_xlabel(label)
    ax.set_ylabel("Score")

    plot_helper('', 'hyper_tune_' + name + '_'+ label, folder=folder, show=False)

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
