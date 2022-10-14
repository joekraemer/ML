from mlrose_hiive import TSPGenerator, QueensGenerator, QueensOpt, KnapsackOpt
import numpy as np
from GenericTester import GenericTester


class KnapSack(GenericTester):
    def __init__(self):
        super().__init__(name='knapsack', complexity_list=range(10, 100, 30))

    def problem_constructor(self, complexity=40, seed=123456):
        np.random.seed(seed)

        weights = np.random.randint(1, 15, complexity)
        values = np.random.randint(1, 10, complexity)
        max_weight_pct = 0.6
        problem = KnapsackOpt(weights=weights,
                              values=values,
                              max_weight_pct=max_weight_pct,
                              maximize=True)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=len(weights))
        return problem, init_state


if __name__ == "__main__":
    tester = KnapSack()
    tester.run_experiment_iterations()
    tester.run()


