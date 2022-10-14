from mlrose_hiive import TSPGenerator, QueensGenerator
import numpy as np
from GenericTester import GenericTester


class Queens(GenericTester):
    def __init__(self):
        super().__init__(name='nqueens', complexity_list=range(30, 120, 30))

    def problem_constructor(self, complexity=70):
        problem = QueensGenerator().generate(seed=123456, size=complexity)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state


if __name__ == "__main__":
    tester = Queens()
    tester.run()


