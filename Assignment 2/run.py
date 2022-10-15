import time
from queens import Queens
from fourpeaks import FourPeaks
from knapsack import KnapSack
from flipflop import FlipFlop
from kcolor import KColor

if __name__ == '__main__':

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

    total_time = time.time() - start
    print("Total Run Time:", total_time)

