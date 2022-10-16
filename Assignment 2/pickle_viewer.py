import pickle
from pathlib import Path
from sys import modules
from os import listdir
from tests.hyperparameter_tester import HyperParamResult

def get_pickle_files(folder_path):
    pickles = []

    dirs = []
    for path in folder_path.iterdir():
        if path.suffix == '.pickle':
            dirs.append(path)

    for file in [path for path in folder_path.iterdir() if path.suffix == '.pickle']:
        with open(file, "rb") as fp:
            object = pickle.load(fp)
            pickles.append(object)
    return pickles


def hyper_table_builder(pk):
    # first just avg evals to converge
    for keys, items in pk.FitnessDict:
        # key is the hyper param
        # item is a list containing the fitness curves for each run

        # for each run in item:
            # find the final fitness score
            # something like run.tail(1)[1]
            # then find the index where the first value where this appeared






if __name__ == "__main__":

    problem = 'onemax'
    root_dir = Path(".")
    path = path = root_dir / 'Graphs' / problem
    # grab all the pickle files in a folder
    pickles = get_pickle_files(path)
    print('Pickles Loaded')

    # I want to look at evaluations to reach convergence
    # Success rate of getting to the global maximum
    # The average wall clock time

    # First lets just build something for average evals to reach convergence

    # we need to grab the correct pickle file...
    for res in pickles:
        if type(res) == HyperParamResult:
            hyper_table_builder()


