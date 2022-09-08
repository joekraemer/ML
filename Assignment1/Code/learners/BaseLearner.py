from abc import ABC, abstractmethod


class BaseLearner(ABC):
    def __init__(self, classifier, verbose=False):
        self._verbose = verbose
        self.Classifier = classifier

    @abstractmethod
    def train(self, X, Y):
        pass

    @abstractmethod
    def test(self, X):
        pass
