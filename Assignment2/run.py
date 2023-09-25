import time
from fourpeaks import FourPeaks
from kcolor import KColor
from onemax import OneMax
from nn_tests import NNTester, NNGridSearchExecutor
from util import loading_data


def run():
    start = time.time()

    ds_red_wine = loading_data.load_red_wine()
    ds_diabetic = loading_data.load_diabetic()

    tester = NNTester()
    tester.run()

    # Run GS on NN algos to try to get somekind of results
    rw_grid_search = NNGridSearchExecutor(ds_red_wine)
    rw_grid_search.run_all_grid_searches()

    tester = NNTester()
    tester.run()

    tester = FourPeaks()
    tester.run()

    tester = KColor()
    tester.run()

    total_time = time.time() - start
    print("Total Run Time:", total_time)


if __name__ == '__main__':
    run()
