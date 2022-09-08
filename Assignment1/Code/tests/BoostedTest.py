from tests.BaseTest import BaseTest
import datetime

from learners.BoostedLearner import BoostedLearner


class BoostedTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='Boosted')
        self._learner = BoostedLearner()

    def run_additional(self, predictions):
        pass
