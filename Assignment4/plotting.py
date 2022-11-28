import glob
import logging
import math
import os
import re
from collections import defaultdict
from os.path import basename
# Configure logging
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import seaborn
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter, FormatStrFormatter
from pandas import DataFrame
from seaborn import PairGrid

from Assignment4.config import A4Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

solvers = ['PI', 'VI', 'QLearning']
environments = ['forest', 'frozen_lake']

MAP_COLORS = {
        'S': 'green',
        'F': 'skyblue',
        'H': 'black',
        'G': 'gold',
    }

MAP_DIR = {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }

MAPS = {
    "4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "12":
        ['SFHFFFFFFFHH',
         'FFFFFFFFFFFF',
         'FFFFFHFFFFHH',
         'FFHFFFFFFFFF',
         'FFFFFFFFFFHF',
         'HFHHHFHFFFFF',
         'FFFFFFFHFFHF',
         'FFHFFFFFHFFF',
         'FFHFFHFFFFFF',
         'FFHFFFHHFFFF',
         'HFFFFHHHFFFF',
         'HFFFFHFFFFHG'],
    "16":
        ['SHFFFFHFFHFFFFHF',
         'FFFFHFFFFFFFHFFF',
         'FFFFFFFFFFFFFHFF',
         'FHFFHFFHFFHHHFFH',
         'FFHFFFFHFHFFHHHF',
         'FHHFFFFFHFHHFHFF',
         'FFFFFFHFFFFFFFHF',
         'FFHFHHFFFFFFFFFF',
         'FFHFFFFFFFFFHFFF',
         'FFFFFFFFFFFFHHFF',
         'HFFHHHFFFHFFFFFF',
         'FHFHFFFFFFFFFFFF',
         'FHHFHFFFFFHHFFFF',
         'FFFFFFFFFFFFHFFF',
         'FFFFFHFFFFHFFFFH',
         'FFFFFHFFFFFFFHFG'],
}

# File name regex to pull dataset name
scree_file_name_regex = re.compile('(.*)_scree\.csv')
multi_scree_file_name_regex = re.compile('(.*)_scree(.*)\.csv')
sse_file_name_regex = re.compile('(.*)_sse\.csv')
acc_file_name_regex = re.compile('(.*)_acc\.csv')
adj_mi_file_name_regex = re.compile('(.*)_adj_mi\.csv')
loglikelihood_file_name_regex = re.compile('(.*)_logliklihood\.csv')
bic_file_name_regex = re.compile('(.*)_bic\.csv')
tsne_file_name_regex = re.compile('(.*)_2D\.csv')
rand_file_name_regex = re.compile('(.*)_rand_index\.csv')
homogeneity_file_name_regex = re.compile('(.*)_homogeneity\.csv')
completeness_file_name_regex = re.compile('(.*)_completeness\.csv')
sil_score_file_name_regex = re.compile('(.*)_sil_score\.csv')
sil_samples_file_name_regex = re.compile('(.*)_sil_samples\.csv')
loss_curve_file_name_regex = re.compile('(.*)_loss_curves\.csv')
train_times_file_name_regex = re.compile('(.*)_train_times\.csv')
learning_curve_file_name_regex = re.compile('(.*)_learning_curve*')
reduced_datasets_name_regex = re.compile('(.*)_dataset\.hdf')
feature_importance_file_name_regex = re.compile('(.*)_feature_importance\.csv')
run_stats_file_name_regex = re.compile('(.*)_scree(.*)\.csv')

algos = {
    'scree': {
        'regex': scree_file_name_regex,
        'descriptive_name': 'Scree'
    },
    'sse': {
        'regex': sse_file_name_regex,
        'descriptive_name': 'SSE'
    },
    'acc': {
        'regex': acc_file_name_regex,
        'descriptive_name': 'Accuracy'
    },
    'adj_mi': {
        'regex': adj_mi_file_name_regex,
        'descriptive_name': 'Adjusted Mutual Information'
    },
    'loglikelihood': {
        'regex': loglikelihood_file_name_regex,
        'descriptive_name': 'Log Likelihood'
    },
    'BIC': {
        'regex': bic_file_name_regex,
        'descriptive_name': 'BIC'
    },
    'sil_score': {
        'regex': sil_score_file_name_regex,
        'descriptive_name': 'Silhouette Score'
    },
    'tsne': {
        'regex': tsne_file_name_regex,
        'descriptive_name': 't-SNE'
    },
    'rand': {
        'regex': rand_file_name_regex,
        'descriptive_name': 'Random Index'
    },
    'homogeneity': {
        'regex': homogeneity_file_name_regex,
        'descriptive_name': 'Homogeneity'
    },
    'completeness': {
        'regex': completeness_file_name_regex,
        'descriptive_name': 'Completeness'
    }
}

WATERMARK = False
GATECH_USERNAME = 'jkraemer6'
TERM = 'Fall 2022'


def watermark(p):
    if not WATERMARK:
        return p

    ax = plt.gca()
    for i in range(1, 11):
        p.text(0.95, 0.95 - (i * (1.0 / 10)), '{} {}'.format(GATECH_USERNAME, TERM), transform=ax.transAxes,
               fontsize=32, color='gray',
               ha='right', va='bottom', alpha=0.2)
    return p


# Adapted from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
def find_knee(values):
    # get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    # np.array([range(nPoints), values])

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    # To calculate the distance to the line, we split vecFromFirst into two
    # components, one that is parallel to the line and one that is perpendicular
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto
    # the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of vecFromFirst onto the line). If we
    # multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def plot_basic(title: str, xlabel: str, ylabel: str, width: int = 4, height: int = 3) -> None:
    plt.close()
    plt.figure(figsize=(width, height))
    # plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    return


def plot_max_utility_iteration(df: DataFrame, xlabel='Iteration', ylabel='Max Utility') -> pyplot:
    plot_basic(title='', xlabel=xlabel, ylabel=ylabel)
    # decide the rolling window size based on the total length of the dataframe
    window = max(1, int(df.shape[0] / 10))

    mean = df['Reward'].rolling(window=window, min_periods=1).mean()
    std = df['Reward'].rolling(window=window, min_periods=1).std()
    plt.plot(df['Iteration'],  mean, color='g')
    # plt.fill_between( df['Iteration'], mean - std, mean + std, alpha=0.1, color="g",)
    return plt


def plot_error_iteration(df: DataFrame, xlabel='Iteration', ylabel='Error') -> pyplot:
    plot_basic(title='', xlabel=xlabel, ylabel=ylabel)

    # decide the rolling window size based on the total length of the dataframe
    window = max(1, int(df.shape[0] / 10))

    if df.shape[0] > 1000:
        data = df['Error'].rolling(window=window, min_periods=1).mean()
        std = df['Error'].rolling(window=window, min_periods=1).std()
    else:
        data = df['Error']
        std = df['Error']

    plt.plot(df['Iteration'], data, color='r')

    temp_df = df.reindex().sort_index(ascending=False)
    last_policy = df.tail(1)['Policy'].values[0]

    last_policy_change = 0
    # determine where the policy last changed
    for i in temp_df.iterrows():
        if i[1].Policy != last_policy:
            last_policy_change = i[1].Iteration + 1
            break

    ax = plt.gca()
    ax.axvline(x=last_policy_change, linestyle="--", label="Last Policy Change: {}".format(last_policy_change))

    plt.legend()
    return plt


def plot_final_policy_frozen_lake(df: DataFrame) -> pyplot:
    size = int(math.sqrt(len(df)))
    try:
        desc = MAPS[str(size)]
    except:
        logger.info("Size of policy doesn't match known map")
        return

    policy = np.reshape(df.to_numpy(), (size, size))

    # handle the things that plot_basic normally would
    plt.close()
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(111, xlim=(0, size), ylim=(0, size))
    font_size = 'x-large'
    if size > 16:
        font_size = 'small'
    for i in range(size):
        for j in range(size):
            y = size - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(MAP_COLORS[desc[i][j]])
            ax.add_patch(p)

            ax.text(x + 0.5, y + 0.5, MAP_DIR[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, size))
    plt.ylim((0, size))

    return plt


def plot_explore_exploit(df: DataFrame, xlabel='Episode', ylabel='Avg Reward') -> pyplot:
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    plot_basic(title='', xlabel=xlabel, ylabel=ylabel)
    # decide the rolling window size based on the total length of the dataframe
    window = max(1, int(df.shape[0] / 20))

    for col in df.columns:
        mean = df[col].rolling(window=window, min_periods=1).mean()
        std = df[col].rolling(window=window, min_periods=1).std()
        color = colors.pop()
        plt.plot(df[col].index, mean, color=color, label="eps={}".format(col))
        plt.fill_between(df[col].index, mean - std, mean + std, alpha=0.1, color=color)

    plt.legend()
    return plt


def save_final_policy_forest(df: DataFrame, problem: str, solver_name: str, output_dir: str) -> None:

    final_policy = list(df.iloc[:, 0])
    save_path = '{}/{}/{}_final_policy.txt'.format(output_dir, problem, solver_name)

    header_string = 'Algorithm & Env & Policy  \\' + '\\'
    line_string = '{} & {} & {} \\'.format(solver_name, problem, final_policy) + '\\'

    if not os.path.exists(save_path):
        lines = [header_string, line_string]
    else:
        lines = [line_string]

    ind = [i for i, value in enumerate(final_policy) if value == 1]
    lines.extend(['', 'Chop Locations', str(ind)])

    _save_lines_to_file(lines, save_path)


def get_solver_name_from_string(s: str) -> str:
    """
    :param s: string (typically a path) that contains solver name
    :return: string of the solver name
    """
    for solver_name in solvers:
        if solver_name in s:
            return solver_name


def read_and_plot_run_stats(problem: str, file: str, output_dir: str) -> None:
    solver_name = get_solver_name_from_string(file)
    logger.info("Plotting run stats for file {} to {} ({})".format(file, output_dir, solver_name))

    df = pd.read_csv(file)

    p = plot_max_utility_iteration(df=df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_reward_iteration.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)

    p = plot_error_iteration(df=df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_error_iteration.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)

    # log the final run times
    save_final_run_times(df, problem, solver_name, output_dir)

    # create a policy gif
    save_policy_gif(df, problem, solver_name, output_dir)

    return


def save_policy_gif(df: DataFrame, problem: str, solver_name: str, output: str) -> None:


    return


def save_final_run_times(df: DataFrame, problem: str, solver_name: str, output_dir: str) -> None:

    total_time = df.tail(1)['Time'].values[-1]
    save_path = '{}/final_run_times.txt'.format(output_dir)

    header_string = 'Algorithm & Env & Time \\' + '\\'
    line_string = '{} & {} & {:.3f} \\'.format(solver_name, problem, total_time) + '\\'

    if not os.path.exists(save_path):
        lines = [header_string, line_string]
    else:
        lines = [line_string]

    _save_lines_to_file(lines, save_path)


def _save_lines_to_file(lines: List[str], path: str):
    with open(path, 'a') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    return


def read_and_plot_final_policy(problem: str, file: str, output_dir: str) -> None:
    solver_name = get_solver_name_from_string(file)
    logger.info("Plotting final policy for file {} to {} ({})".format(file, output_dir, solver_name))

    df = pd.read_csv(file)
    if problem == 'frozen_lake':
        p = plot_final_policy_frozen_lake(df=df)
        p = watermark(p)
        p.savefig('{}/{}/{}_final_policy.png'.format(output_dir, problem, solver_name),
                  format='png', bbox_inches='tight', dpi=250)
    else:
        save_final_policy_forest(df, problem, solver_name, output_dir)

    return


def read_and_plot_explore_exploit(problem: str, file: str, output_dir: str) -> None:
    solver_name = get_solver_name_from_string(file)
    logger.info("Plotting run explore exploit experiment for file {} to {} ({})".format(file, output_dir, solver_name))

    df = pd.read_csv(file, index_col=0)

    p = plot_explore_exploit(df=df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_explore_exploit_reward.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)


def read_and_plot_problem(env_name: str, output_dir: str, input_dir: str, cfg: A4Config):
    env_data_path = '{}/{}'.format(input_dir, env_name)

    out = '{}/{}'.format(output_dir, env_name)
    if not os.path.exists(out):
        os.makedirs(out)

    run_stats_files = glob.glob('{}/**/run_stats.csv'.format(env_data_path), recursive=True)
    logger.info("Results files {}".format(run_stats_files))
    [read_and_plot_run_stats(env_name, f, output_dir) for f in run_stats_files]

    final_policy_files = glob.glob('{}/**/final_policy.csv'.format(env_data_path), recursive=True)
    logger.info("Results files {}".format(final_policy_files))
    [read_and_plot_final_policy(env_name, f, output_dir) for f in final_policy_files]

    exp_files = glob.glob('{}/**/explore_exploit_reward.csv'.format(env_data_path), recursive=True)
    logger.info("Results files {}".format(exp_files))
    [read_and_plot_explore_exploit(env_name, f, output_dir) for f in exp_files]


def plot_results(cfg: A4Config):
    if cfg.plot_old_results:
        logger.info("Processing Previous Experiment {}".format(cfg.old_results_path))

        output_dir = '{}{}'.format(cfg.old_results_path, cfg.paths.plots)
        input_dir = '{}{}'.format(cfg.old_results_path, cfg.paths.results_data)

    else:
        output_dir = cfg.paths.plots
        input_dir = cfg.paths.results_data

    for env_name in environments:
        logger.info("Processing {}".format(env_name))

        read_and_plot_problem(env_name, output_dir=output_dir, input_dir=input_dir, cfg=cfg)


if __name__ == '__main__':
    plot_results()
