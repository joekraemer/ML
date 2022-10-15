from pathlib import Path
import pandas as pd
import numpy as np

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

    evals_dict['rhc'] = pd.DataFrame(evals_dict['rhc'])
    evals_dict['sa'] = pd.DataFrame(evals_dict['sa'])
    evals_dict['ga'] = pd.DataFrame(evals_dict['ga'])
    evals_dict['mimic'] = pd.DataFrame(evals_dict['mimic'])

    times_dict['rhc'] = pd.DataFrame(times_dict['rhc']).mean()
    times_dict['sa'] = pd.DataFrame(times_dict['sa']).mean()
    times_dict['ga'] = pd.DataFrame(times_dict['ga']).mean()
    times_dict['mimic'] = pd.DataFrame(times_dict['mimic']).mean()

    lines = ["rhc & total evaluations & wall clock time (sec) & time/eval (ms) & time/iteration (ms)"]

    for key, avg_time in times_dict.items():
        # since I don't trust evals to restart, I have to compute my own for each row of the df
        diffed = evals_dict[key].diff(axis=1)
        cummed_axis_1_T = diffed.cumsum(axis=1).transpose()

        # Need to grab the last number in each series
        # ffill the last number to the end of the dataframe?
        ffilled = cummed_axis_1_T.ffill()
        avg_total_evals = float(ffilled.tail(1).mean(axis=1))

        # number of iterations
        total_iterations = evals_dict[key].count().sum()
        avg_total_iterations = total_iterations/evals_dict[key].shape[0]

        time_per_eval = (float(avg_time)/avg_total_evals) * 1000
        time_per_iteration = (float(avg_time)/avg_total_iterations) * 1000

        line = key + " & " + str(int(avg_total_evals)) + " & " + f'{float(avg_time):.2f}' + \
               " & " +  f'{time_per_eval:.2f}' +  " & " + f'{time_per_iteration:.2f}' + " \\" + "\\"

        lines.append(line)
    # what I want is a table
    # rhc & total evaluations & wall clock time & time/eval ms & time/iteration ms

    save_table_to_file(lines, name, 'evals_table')
    return


def logging_scoring_metrics(lines, folder, name):

    save_table_to_file(lines, folder=folder, filename= name + '_gs_scoring_metrics_')



def save_table_to_file(lines, folder, filename):
    root_dir = Path(".")
    path = root_dir / 'Graphs' / folder / (filename + ".txt")

    with open(path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    return


