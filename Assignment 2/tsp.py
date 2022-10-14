from mlrose_hiive import TSPGenerator
import numpy as np
from GenericTester import GenericTester


class TSP(GenericTester):
    def __init__(self):
        super().__init__(name='TSP')

    def problem_constructor(self, complexity=20):
        problem = TSPGenerator().generate(seed=123456, number_of_cities=complexity)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state


if __name__ == "__main__":
    tester = TSP()
    tester.run()


