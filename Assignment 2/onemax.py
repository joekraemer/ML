import mlrose_hiive
from mlrose_hiive import FlipFlopGenerator
import numpy as np
from GenericTester import GenericTester


class OneMax(GenericTester):
    def __init__(self, debug=False):
        if debug:
            complexity_list = range(2, 20, 5)
        else:
            complexity_list = range(10, 100, 30)

        hyper_tuning = {'ga_pop_size': [2, 5, 10, 40],
                        'ga_default_pop': 10,
                        'mimic_pop_size': [50, 100, 300, 400]}
        super().__init__(name='onemax', complexity_list=complexity_list, debug=debug, hyper_config=hyper_tuning)

        if debug:
            print("Debug mode")
            self.Complexity = 30
        else:
            self.Complexity = 50
        return

    def problem_constructor(self, complexity=None, seed=123456):
        if complexity is None:
            complexity = self.Complexity
        fitness = mlrose_hiive.OneMax()
        problem = mlrose_hiive.DiscreteOpt(length=complexity, fitness_fn=fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.random_hill_climb(problem, max_attempts=max_attempts, max_iters=2000, restarts=10,
                                                                init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                         max_attempts=max_attempts, max_iters=2000,
                                         init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        return mlrose_hiive.genetic_alg(problem, pop_size=50, mutation_prob=0.2, max_attempts=500, max_iters=2000, curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        return mlrose_hiive.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=500, max_iters=2000,
                           curve=curve)

    def run_extra(self):
        # Run a test where GA uses uniform crossover instead
        return


if __name__ == "__main__":
    tester = OneMax(debug=True)
    tester.run()


