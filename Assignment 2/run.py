import time
from queens import Queens
from fourpeaks import FourPeaks
from knapsack import KnapSack
from flipflop import FlipFlop
from kcolor import KColor
from nn_tests import NNTester, NNGridSearchExecutor
from util import loading_data


def run():
    start = time.time()

    tester = FourPeaks()
    tester.run()

    tester = KColor()
    tester.run()

    tester = FlipFlop()
    tester.run()

    tester = KnapSack()
    tester.run()

    tester = Queens()
    tester.run()

    tester = NNTester()
    tester.run()

    ds_red_wine = loading_data.load_red_wine()
    ds_diabetic = loading_data.load_diabetic()

    # Run GS on NN algos to try to get somekind of results
    rw_grid_search = NNGridSearchExecutor(ds_red_wine)
    rw_grid_search.run_all_grid_searches()

    total_time = time.time() - start
    print("Total Run Time:", total_time)


if __name__ == '__main__':
    run()
