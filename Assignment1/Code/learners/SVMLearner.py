from learners.BaseLearner import BaseLearner

from sklearn import svm


class SVMLearner(BaseLearner):
    def __init__(self, C=1, kernel='rbf'):
        super().__init__(svm.SVC(C=C, kernel=kernel))

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
