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

    ds_red_wine = loading_data.load_red_wine()
    ds_diabetic = loading_data.load_diabetic()

    tester = NNTester()
    tester.run()

    # Run GS on NN algos to try to get somekind of results
    rw_grid_search = NNGridSearchExecutor(ds_red_wine)
    rw_grid_search.run_all_grid_searches()

    tester = Queens()
    tester.run()

    tester = KnapSack()
    tester.run()

    tester = NNTester()
    tester.run()

    tester = FourPeaks()
    tester.run()

    tester = KColor()
    tester.run()

    tester = FlipFlop()
    tester.run()

    total_time = time.time() - start
    print("Total Run Time:", total_time)


if __name__ == '__main__':
    run()
