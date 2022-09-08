from learners.BaseLearner import BaseLearner

from sklearn import svm


class SVMLearner(BaseLearner):
    def __init__(self, n_neighbors=5, algorithm='ball_tree'):
        super().__init__(svm.SVC())

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
