import mlrose_hiive
from mlrose_hiive import TSPGenerator, QueensGenerator, QueensOpt, KnapsackOpt, KnapsackGenerator
import numpy as np
from GenericTester import GenericTester


class KnapSack(GenericTester):
    def __init__(self, debug=False):
        knapsack_hyper_tuning = {'ga_pop_size': [100, 300, 500],
                                 'mimic_pop_size': [100, 300, 500]}
        super().__init__(name='knapsack', complexity_list=range(50, 250, 50), debug=debug,
                         hyper_config=knapsack_hyper_tuning)

    def problem_constructor(self, complexity=200, seed=123456):
        problem = KnapsackGenerator().generate(number_of_items_types=complexity,
                                               max_item_count=100, max_weight_per_item=40,
                                               max_value_per_item=20, max_weight_pct=0.6, seed=seed)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):
        return mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=20000, restarts=10,
                                              init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                max_attempts=1000, max_iters=20000,
                                                init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        return mlrose_hiive.genetic_alg(problem, max_attempts=500, max_iters=20000, curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        return mlrose_hiive.mimic(problem, pop_size=500, max_attempts=500, max_iters=20000,
                                  curve=curve)


if __name__ == "__main__":
    tester = KnapSack()
    tester.run_experiment_iterations()
    tester.run()
