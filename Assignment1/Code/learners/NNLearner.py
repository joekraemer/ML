from learners.BaseLearner import BaseLearner

from sklearn.neural_network import MLPClassifier


class NNLearner(BaseLearner):
    def __init__(self, alpha=1e-5, n_nodes=50, n_layers=3):
        hidden_layers = tuple(n_nodes for _ in range(n_layers))
        super().__init__(MLPClassifier(solver='sgd', alpha=alpha,
                                            hidden_layer_sizes=hidden_layers, random_state=1, max_iter=5000))

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
