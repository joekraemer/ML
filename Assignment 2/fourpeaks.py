import mlrose_hiive
from mlrose_hiive import FlipFlopGenerator, OnePointCrossOver
import numpy as np
from GenericTester import GenericTester


class FourPeaks(GenericTester):
    def __init__(self, debug=False):
        hyper_tuning = {'ga_pop_size': [50, 200, 300, 400],
                                 'mimic_pop_size': [50, 200, 300, 400]}
        super().__init__(name='four_peaks', complexity_list=range(30, 100, 20), debug=debug,
                         hyper_config=hyper_tuning)

    def problem_constructor(self, complexity=70, seed=123456):
        fitness = mlrose_hiive.FourPeaks(t_pct=0.2)
        problem = mlrose_hiive.DiscreteOpt(length=complexity, fitness_fn=fitness, maximize=True, max_val=2)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):
        return mlrose_hiive.random_hill_climb(problem, max_attempts=1000, max_iters=20000, restarts=10,
                                                                init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(),
                                         max_attempts=500, max_iters=20000,
                                         init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        return mlrose_hiive.genetic_alg(problem, pop_size=300, mutation_prob=0.2, max_attempts=500, max_iters=20000, curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        return mlrose_hiive.mimic(problem, pop_size=300, keep_pct=0.2, max_attempts=500, max_iters=20000,
                           curve=curve)

    def run_extra(self):
        # I want to run 4peaks while varying the size of the attraction basins
        # TODO
        return


if __name__ == "__main__":
    tester = FourPeaks()
    tester.run()


