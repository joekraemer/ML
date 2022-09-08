from tests.BaseTest import BaseTest
import datetime

from learners.SVMLearner import SVMLearner


class SVMTests(BaseTest):
    def __init__(self, details):
        super().__init__(details, name='SVM')
        self._learner = SVMLearner()

    def run_additional(self, predictions):
        pass
