from matplotlib import pyplot
from sklearn import svm
import numpy as np

class SVM:
    def __init__(self, X, y):
        self.model = svm.SVC( kernel='linear', cache_size=2000)
        self.model.fit(X, y)
        self.importance_feature = self.model.coef_

    def plot_importance_feature(self):
        for i, v in enumerate(self.importance_feature):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(self.importance_feature))], self.importance_feature)
        pyplot.savefig('./data/fig/svm-if.png')
