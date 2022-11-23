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

FROZEN_LAKE_DESC = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

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


def plot_reward_iteration(df: DataFrame, xlabel='Iteration', ylabel='Reward') -> pyplot:
    plot_basic(title='', xlabel=xlabel, ylabel=ylabel)

    plt.plot(df['Iteration'], df['Reward'], 'o-', linewidth=1, markersize=2)
    return plt


def plot_error_iteration(df: DataFrame, xlabel='Iteration', ylabel='Error') -> pyplot:
    plot_basic(title='', xlabel=xlabel, ylabel=ylabel)

    plt.plot(df['Iteration'], df['Error'], 'o-', linewidth=1, markersize=2)
    return plt


def plot_final_policy_frozen_lake(df: DataFrame) -> pyplot:
    size = int(math.sqrt(len(df)))
    if size != len(FROZEN_LAKE_DESC):
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
            p.set_facecolor(MAP_COLORS[FROZEN_LAKE_DESC[i][j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, MAP_DIR[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, size))
    plt.ylim((0, size))

    return plt


def plot_final_policy_forest(df: DataFrame ) -> pyplot:
    pass


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

    p = plot_reward_iteration(df=df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_reward_iteration.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)

    p = plot_error_iteration(df=df)
    p = watermark(p)
    p.savefig(
        '{}/{}/{}_error_iteration.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)

    return

def read_and_plot_final_policy(problem: str, file: str, output_dir: str) -> None:
    solver_name = get_solver_name_from_string(file)
    logger.info("Plotting final policy for file {} to {} ({})".format(file, output_dir, solver_name))

    df = pd.read_csv(file)
    if problem == 'frozen_lake':
        p = plot_final_policy_frozen_lake(df=df)
    else:
        p = plot_final_policy_forest(df=df)

    p = watermark(p)
    p.savefig('{}/{}/{}_final_policy.png'.format(output_dir, problem, solver_name),
        format='png', bbox_inches='tight', dpi=250)

    return


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
