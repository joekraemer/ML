from learners.BaseLearner import BaseLearner

from sklearn.neural_network import MLPClassifier


class NNLearner(BaseLearner):
    def __init__(self, alpha=1e-5, n_nodes=20):
        super().__init__(MLPClassifier(solver='lbfgs', alpha=alpha,
                                            hidden_layer_sizes=(n_nodes, n_nodes), random_state=1))

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
