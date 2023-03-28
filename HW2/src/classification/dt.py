from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier


class DT:
    def __init__(self, X, y):
        # define the model
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)
        self.importance_feature = self.model.feature_importances_

    def plot_importance_feature(self):
        for i, v in enumerate(self.importance_feature):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(self.importance_feature))], self.importance_feature)
        pyplot.savefig('./data/fig/dt-if.png')
