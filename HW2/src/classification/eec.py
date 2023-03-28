from imblearn.ensemble import EasyEnsembleClassifier
from matplotlib import pyplot
from sklearn.inspection import permutation_importance


class EEC:
    def __init__(self, X, y):
        # define the model
        self.model = EasyEnsembleClassifier()
        self.model.fit(X, y)
        results = permutation_importance(self.model, X, y, scoring='neg_mean_squared_error')
        self.importance_feature = results.importances_mean
        self.model.feature_importances_ = self.importance_feature

    def plot_importance_feature(self):
        for i, v in enumerate(self.importance_feature):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(self.importance_feature))], self.importance_feature)
        pyplot.savefig('./data/fig/eec-if.png')