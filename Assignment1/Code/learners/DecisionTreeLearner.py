from learners.BaseLearner import BaseLearner

from sklearn import tree


class DecisionTreeLearner(BaseLearner):
    def __init__(self, leaf_size=1, ccp_alpha=0.0, max_depth=6):
        super().__init__(tree.DecisionTreeClassifier(min_samples_leaf=leaf_size, ccp_alpha=ccp_alpha, max_depth=max_depth))

    def train(self, X, Y):
        self.Classifier.fit(X, Y)
        pass

    def test(self, X):
        prediction = self.Classifier.predict(X)
        return prediction
