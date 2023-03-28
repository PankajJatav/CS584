from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression


class Logistic:
    def __init__(self, X, y):
        self.model = LogisticRegression()
        # fit the model
        self.model.fit(X, y)
        # get importance
        self.importance_feature = self.model.coef_[0]

    def plot_importance_feature(self):
        for i, v in enumerate(self.importance_feature):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(self.importance_feature))], self.importance_feature)
        pyplot.savefig('./data/fig/lr-if.png')