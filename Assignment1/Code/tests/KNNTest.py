from tests.BaseTest import BaseTest
import datetime

from learners.KNNLearner import KNNLearner


class KNNTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='KNN')
        self._learner = KNNLearner()

    def run_additional(self, predictions):
        pass
