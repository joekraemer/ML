from pathlib import Path
import pandas as pd
import numpy as np
import pickle

lines = ['Readme', 'How to write text files in Python']


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

    temp_evals_dict['rhc'] = pd.DataFrame(evals_dict['rhc'])
    temp_evals_dict['sa'] = pd.DataFrame(evals_dict['sa'])
    temp_evals_dict['ga'] = pd.DataFrame(evals_dict['ga'])
    temp_evals_dict['mimic'] = pd.DataFrame(evals_dict['mimic'])

    temp_times_dict['rhc'] = pd.DataFrame(times_dict['rhc']).mean()
    temp_times_dict['sa'] = pd.DataFrame(times_dict['sa']).mean()
    temp_times_dict['ga'] = pd.DataFrame(times_dict['ga']).mean()
    temp_times_dict['mimic'] = pd.DataFrame(times_dict['mimic']).mean()

    lines = ["rhc & total evaluations & wall clock time (sec) & time/eval (ms) & time/iteration (ms) & evals/iteration"]

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

        time_per_eval = (float(avg_time) / avg_total_evals) * 1000
        time_per_iteration = (float(avg_time) / avg_total_iterations) * 1000

        avg_evals_per_iteration = avg_total_evals / avg_total_iterations

        line = key + " & " + str(int(avg_total_evals)) + " & " + f'{float(avg_time):.2f}' + \
               " & " + f'{time_per_eval:.2f}' + " & " + f'{time_per_iteration:.2f}' + " & " \
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


def path_builder(dataset, filename, folder='Graphs'):
    root_dir = Path(".")
    path = root_dir / folder / dataset / (filename)
    return path


def load_obj_as_pickle(filepath):
    with open(filepath, 'rb') as f:
        object = pickle.load(f)

    return object
