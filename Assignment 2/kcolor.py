import mlrose_hiive
import numpy as np
from mlrose_hiive import MaxKColorOpt
import networkx as nx
from GenericTester import GenericTester


class MaxKColorGenerator:
    @staticmethod
    def generate(seed, number_of_nodes=20, max_connections_per_node=4, max_colors=None, maximize=False):

        """
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        """
        np.random.seed(seed)
        # all nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                      n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = MaxKColorOpt(edges=edges, length=number_of_nodes, maximize=maximize, max_colors=max_colors, source_graph=g)
        return problem


class KColor(GenericTester):
    def __init__(self, debug=False):
        hyper_tuning = {'ga_pop_size': [50, 100, 200],
                                 'mimic_default_keep': 0.3,
                                 'mimic_default_pop': 200,
                                 'mimic_pop_size': [100, 300, 500],
                                 'mimic_keep_percent': [0.15, 0.3, 0.5]}

        super().__init__(name='kcolors', complexity_list=range(10, 50, 20), hyper_config=hyper_tuning, debug=debug)

    def problem_constructor(self, complexity=20, seed=123456):
        max_connections = 4 # typically 1/5
        problem = MaxKColorGenerator().generate(seed=123456, number_of_nodes=complexity, max_connections_per_node=max_connections, maximize=True)
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=complexity)
        return problem, init_state

    def run_best_rhc(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.random_hill_climb(problem, max_attempts=max_attempts, max_iters=20000, restarts=10,
                                              init_state=init_state, curve=curve)

    def run_best_sa(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(),
                                                max_attempts=max_attempts, max_iters=20000,
                                                init_state=init_state, curve=curve)

    def run_best_ga(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.genetic_alg(problem, max_attempts=max_attempts, max_iters=10000, curve=curve)

    def run_best_mimic(self, problem, init_state, curve=True):
        max_attempts = self.calc_max_attempts(problem.length)
        return mlrose_hiive.mimic(problem, keep_pct=0.5, pop_size=200, max_attempts=max_attempts, max_iters=10000,
                                  curve=curve)

    def run_extra(self):
        #TODO: - how does changing the number of connections affect the performance
        return


if __name__ == "__main__":
    tester = KColor(debug=True)
    tester.run_hyperparameters()
    tester.run()

