from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class RF:
    def __init__(self, X, y):
        self.model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        # RandomForestClassifier(random_state=0)
        self.model.fit(X, y)
        self.importance_feature = self.model.feature_importances_

    def plot_importance_feature(self):
        for i, v in enumerate(self.importance_feature):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(self.importance_feature))], self.importance_feature)
        pyplot.savefig('./data/fig/rf-if.png')

