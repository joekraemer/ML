from mlrose_hiive import MaxKColorGenerator
import numpy as np
from GenericTester import GenericTester


class KColor(GenericTester):
    def __init__(self):
        super().__init__(name='flipflop', complexity_list=range(5, 20, 5))

    def problem_constructor(self, complexity=20, seed=123456):
        problem = MaxKColorGenerator().generate(seed=123456, number_of_nodes=complexity)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state


if __name__ == "__main__":
    tester = KColor()
    tester.run()


