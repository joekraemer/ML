from learners.BaseLearner import BaseLearner

from sklearn.ensemble import AdaBoostClassifier


class BoostedLearner(BaseLearner):
    def __init__(self, n_estimators=65, learning_rate=1, base_estimator=None):
        super().__init__(AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, base_estimator=base_estimator))

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
