import pickle
from pathlib import Path

import pandas as pd

lines = ['Readme', 'How to write text files in Python']


def log_hyper_table(evals_dict, fitness_dict, times_dict, folder, name):
    """
    evaluations_dict = {
            "rhc": [ndarray(), ndarray(), ... ],
            "sa": [],
            ...
        }

    fitness_dict = {
        "rhc": [ndarray(), ndarray(), ... ],
        "sa": [],
        ...
        }

    times_dict = {
            "rhc": [],
            "sa": [],
            ...
        }
    """
    temp_evals_dict = {}
    temp_fitness_dict = {}
    temp_times_dict = {}

    for key, item in evals_dict.items():
        temp_evals_dict[key] = pd.DataFrame(item)

    for key, item in times_dict.items():
        temp_times_dict[key] = pd.DataFrame(item).mean()

    for key, item in fitness_dict.items():
        temp_fitness_dict[key] = pd.DataFrame(item)

    lines = [
        " Hyperparameter & Total Fxn Evals & Run Time (sec) & Time / Iterations (ms) & Fxn Evals / Iteration & Avg Fitness & Fitness Std  \\"]

    for key, avg_time in temp_times_dict.items():
        # since I don't trust evals to restart, I have to compute my own for each row of the df
        diffed = temp_evals_dict[key].diff(axis=1)
        cummed_axis_1_T = diffed.cumsum(axis=1).transpose()

        # Need to grab the last number in each series
        # ffill the last number to the end of the dataframe?
        ffilled = cummed_axis_1_T.ffill()
        avg_total_evals = float(ffilled.tail(1).mean(axis=1))

        # number of iterations
        total_iterations = temp_evals_dict[key].count().sum()
        avg_total_iterations = total_iterations / temp_evals_dict[key].shape[0]

        time_per_iteration = (float(avg_time) / avg_total_iterations) * 1000

        avg_evals_per_iteration = avg_total_evals / avg_total_iterations

        # whats the last fitness value?
        ffilled = temp_fitness_dict[key].transpose().ffill()
        avg_fitness = float(ffilled.tail(1).mean(axis=1))
        std_fitness = float(ffilled.tail(1).std(axis=1))

        line = key + " & " + str(int(avg_total_evals)) + " & " + f'{float(avg_time):.2f}' + \
               " & " + f'{time_per_iteration:.2f}' + " & " \
               + f'{avg_evals_per_iteration:.1f}' + " & " + f'{avg_fitness:.2f}' + " & " + f'{std_fitness:.2f}' + " \\" + "\\"

        lines.append(line + "\\")
    # what I want is a table
    # rhc & total evaluations & wall clock time & time/eval ms & time/iteration ms

    save_table_to_file(lines, folder, name)
    return


def log_evals_table(evals_dict, times_dict, name):
    """
    evaluations_dict = {
            "rhc": [ndarray(), ndarray(), ... ],
            "sa": [],
            ...
        }

    times_dict = {
            "rhc": [],
            "sa": [],
            ...
        }
    """
    temp_evals_dict = {}
    temp_times_dict = {}

    for key, item in evals_dict.items():
        temp_evals_dict[key] = pd.DataFrame(item)

    for key, item in times_dict.items():
        temp_times_dict[key] = pd.DataFrame(item).mean()

    lines = [" Algorithm & Total Fxn Evals & Convergence Time (sec) & Time / Iterations (ms) & Fxn Evals/Iteration \\"]

    for key, avg_time in temp_times_dict.items():
        # since I don't trust evals to restart, I have to compute my own for each row of the df
        diffed = temp_evals_dict[key].diff(axis=1)
        cummed_axis_1_T = diffed.cumsum(axis=1).transpose()

        # Need to grab the last number in each series
        # ffill the last number to the end of the dataframe?
        ffilled = cummed_axis_1_T.ffill()
        avg_total_evals = float(ffilled.tail(1).mean(axis=1))

        # number of iterations
        total_iterations = temp_evals_dict[key].count().sum()
        avg_total_iterations = total_iterations / temp_evals_dict[key].shape[0]

        time_per_iteration = (float(avg_time) / avg_total_iterations) * 1000

        avg_evals_per_iteration = avg_total_evals / avg_total_iterations

        line = key + " & " + str(int(avg_total_evals)) + " & " + f'{float(avg_time):.2f}' + \
               " & " + f'{time_per_iteration:.2f}' + " & " \
               + f'{avg_evals_per_iteration:.1f}' + " \\" + "\\"

        lines.append(line + "\\")
    # what I want is a table
    # rhc & total evaluations & wall clock time & time/eval ms & time/iteration ms

    save_table_to_file(lines, name, 'evals_table')
    return


def logging_scoring_metrics(lines, folder, name):
    save_table_to_file(lines, folder=folder, filename=name + '_gs_scoring_metrics_')
    return


def save_table_to_file(lines, folder, filename):
    root_dir = Path(".")
    path = root_dir / 'Graphs' / folder / (filename + ".txt")

    with open(path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    return


def save_obj_as_pickle(object, folder, filename):
    root_dir = Path(".")
    path = root_dir / 'Graphs' / folder / (filename + ".pickle")

    # Now just pickle this full_res object and worry about graphing and logging later
    with open(path, 'wb') as f:
        pickle.dump(object, f)

    return
