from tests.BaseTest import BaseTest
import datetime

from learners.NNLearner import NNLearner


class NNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='NN')
        self._learner = NNLearner()

    def run_additional(self, predictions):
        pass
