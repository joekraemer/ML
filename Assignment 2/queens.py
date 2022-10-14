import mlrose_hiive
from mlrose_hiive import TSPGenerator, QueensGenerator, QueensOpt
import numpy as np
from GenericTester import GenericTester


class Queens(GenericTester):
    def __init__(self):
        super().__init__(name='nqueens', complexity_list=range(4, 50, 20))

    def problem_constructor(self, complexity=20, seed=123456):
        np.random.seed(seed)
        problem = QueensOpt(length=complexity, maximize=True)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):
        return mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=20000, restarts=10,
                                              init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
                                                max_attempts=1000, max_iters=20000,
                                                init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        return mlrose_hiive.genetic_alg(problem, pop_size=200, mutation_prob=0.2, max_attempts=500, max_iters=20000,
                                        curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        return mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=500, max_iters=20000,
                                  curve=curve)


if __name__ == "__main__":
    tester = Queens()
    tester.run_experiment_iterations()
    tester.run()
