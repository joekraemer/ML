
import time
import gym
import numpy as np
from math import sqrt
from gym.envs import toy_text

from Assignment4.src.solvers.config import QLearningConfig
import hiive.mdptoolbox.util as _util


class QLearner(object):
    """Adapted from my Deep Learning class and modified to match the api and output of the MDP toolbox"""
    def __init__(self, env_wrapper, cfg: QLearningConfig):
        self.Q = None
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.epsilon = 1
        self.eps_decay = cfg.epsilon_decay
        self.eps_end = cfg.epsilon_min

        self.MaxEpisodes = 100000

        self.env = env_wrapper.env

        self.NumStates = self.env.observation_space.n
        self.NumActions = self.env.action_space.n

        self.ConvergenceValue = cfg.convergence_value
        self.run_stats = None

    def run(self):
        """run the agent through the environment"""
        self.Q = np.zeros((self.NumStates, self.NumActions))
        self.V = self.Q.max(axis=1)
        run_stats = []

        for e in range(0, self.MaxEpisodes):

            s, info = self.env.reset()
            Qprev = self.Q.copy()

            self.time = time.time()
            done = False
            while not done:
                a = self.select_action(s, self.Q, self.epsilon, self.env)
                s_prime, r, done, truncated, info = self.env.step(a)
                a_prime = self.select_greedy_action(s_prime, self.Q)

                # q learning update function
                # Q(s_t,a_t) = Q(s_t,a_t) + alpha * (r_t + gamma * (Q(s_t+1, a_t+1)))
                update_component = self.alpha * (
                        r + self.gamma * self.Q[s_prime, a_prime] - self.Q[s, a])

                self.Q[s, a] += update_component

                error = np.absolute(update_component)
                s = s_prime

            # compute the value function and the policy
            v = self.Q.max(axis=1)
            self.V = v
            p = self.Q.argmax(axis=1)
            self.policy = p
            error = _util.getSpan(self.Q - Qprev)

            run_stats.append(self._build_run_stat(i=e, s=s, a=None, r=np.max(self.V), p=p, v=v, error=error))

            if error < self.ConvergenceValue:
                break

            if self.epsilon > self.eps_end:
                self.epsilon = self.epsilon * self.eps_decay

        self.env.close()

        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats

    def select_action(self, _s, _q, _eps, env):
        # an action should be selected uniformly at random if a random number drawn uniformly
        # between 0 and 1 is less than e. If the greedy action is selected, the action with lowest index
        # should be selected in case of ties
        # greedy check
        res = np.random.rand()
        if res < _eps:
            return np.random.randint(env.action_space.n)
        else:
            return self.select_greedy_action(_s, _q)

    def select_greedy_action(self, _s, _q):
        # best action selection
        return np.argmax(_q[_s, :])

    def Q_table(self, state, action):
        """return the optimal value for State-Action pair in the Q Table"""
        return self.Q[state][action]

    def _build_run_stat(self, i, a, error, p, r, s, v):
        run_stat = {
            'State': s,
            'Action': a,
            'Reward': r,
            'Error': error,
            'Time': time.time() - self.time,
            'Alpha': self.alpha,
            'Epsilon': self.epsilon,
            'Gamma': self.gamma,
            'V[0]': v[0],
            'Max V': np.max(v),
            'Mean V': np.mean(v),
            'Iteration': i,
            # 'Value': v.copy(),
            'Policy': p.copy()
        }
        return run_stat
