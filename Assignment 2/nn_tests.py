import numpy as np
import pandas as pd
import mlrose_hiive
import matplotlib.pyplot as plt
import time
import datetime
import os

from util import loading_data
from util.graphing import plot_learning_curve, plot_scalability, plot_single_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate, KFold

# use randomized hill climbing, simulated annealing, and genetic algorithm to find good weights for a neural network

N_JOBS = 4
NN_STRUCTURE = [50, 50, 50]
MAX_ITERATION = 1000


def exp_loss_curve(ds, backprop_algo='gradient_descent'):
    loss_curves = []

    fold_iterator = StratifiedShuffleSplit(n_splits=12, test_size=0.1, random_state=0)
    # fold_iterator = KFold(n_splits=3)

    temp_learner = mlrose_hiive.NeuralNetwork(hidden_nodes=NN_STRUCTURE, algorithm=backprop_algo, max_iters=1000, learning_rate=1e-5, curve=True)
    res = cross_validate(
        temp_learner,
        ds.train_x,
        ds.train_y,
        scoring="accuracy",
        cv=fold_iterator,
        n_jobs=N_JOBS,
        return_train_score=True,
        return_estimator=True
    )

    for e in res['estimator']:
        loss_curves.append(e.fitness_curve)

    loss_curves_df = pd.DataFrame(loss_curves)
    plot_single_curve(loss_curves_df, 'NN_loss_curve_' + backprop_algo, '', 'Loss', 'iteration', folder='nn_' + ds.name)


def exp_scalability(ds, backprop_algo='gradient_descent'):
    # TODO: maybe multiple algos can be combined on one chart
    learner = mlrose_hiive.NeuralNetwork(hidden_nodes=NN_STRUCTURE, algorithm=backprop_algo, curve=True)
    train_sizes = np.linspace(0.1, 1.0, 10)
    validation_fold_iterator = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        learner,
        ds.train_x,
        ds.train_y,
        scoring="accuracy",
        cv=validation_fold_iterator,
        n_jobs=N_JOBS,
        train_sizes=train_sizes,
        return_times=True,
    )

    fit_times_ms = fit_times * 1000

    plot_learning_curve(train_scores, test_scores, train_sizes, backprop_algo, folder='nn_' + ds.name)
    plot_scalability(fit_times_ms, train_sizes, backprop_algo, folder='nn_' + ds.name)
    return


def run_nn():
    seed = 123456
    print("Seed {}".format(seed))

    print("Load datasets\r\n")

    ds_red_wine = loading_data.load_red_wine()
    ds_diabetic = loading_data.load_diabetic()

    print("Starting Tests....")

    exp_loss_curve(ds_red_wine)
    exp_scalability(ds_red_wine)

    exp_loss_curve(ds_diabetic)
    exp_scalability(ds_diabetic)

    return


if __name__ == "__main__":
    seed = 123456
    print("Seed {}".format(seed))

    run_nn()