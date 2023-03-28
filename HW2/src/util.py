import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import RandomOverSampler

from classification.RF import RF
from classification.dt import DT
from classification.eec import EEC
from classification.knn import KNN
from classification.logistic import Logistic
from classification.nbayes import NBayes
from classification.svm import SVM



def load_data(file_loc, isRemoveD = False):
    df = pd.read_csv(file_loc)
    df.columns = df.columns.str.replace(' ', '')
    df['F10'] = df['F10'].str.strip()
    df['F11'] = df['F11'].str.strip()

    if (isRemoveD):
        df = df.drop_duplicates(df.columns[1:])

    df['F10'] = df['F10'].rank(method='dense', ascending=False).astype(int)
    df['F11'] = df['F11'].rank(method='dense', ascending=False).astype(int)

    df = df[df.columns[1:]]

    # Remove duo to low accuracy
    # x = df.values
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # return pd.DataFrame(x_scaled)
    return df

def predict(modelObj, ModelClass, X_train, X_test, y_train,  feature_selection ):
    if feature_selection:
        threshold = modelObj.importance_feature.min() + (modelObj.importance_feature.max() + modelObj.importance_feature.min()) * 0.1
        model = SelectFromModel(modelObj.model, prefit=True, threshold=threshold)
        X_new = model.transform(X_train)
        modelObj = ModelClass(X_new, y_train)
        X_test = model.transform(X_test)

    return modelObj.model.predict(X_test)

def predict_model(type, X_train, y_train, X_test, feature_selection = False):

    if type == 'knn':
        knn = KNN(X_train, y_train)
        return predict(knn, KNN, X_train, X_test, y_train,  feature_selection)

    if type == 'svm':
        svm = SVM(X_train, y_train)
        return predict(svm, SVM, X_train, X_test, y_train,  feature_selection)

    if type == 'rf':
        rf = RF(X_train, y_train)
        return predict(rf, RF, X_train, X_test, y_train,  feature_selection)

    if type == 'logistic':
        logistic = Logistic(X_train, y_train)
        return predict(logistic, Logistic, X_train, X_test, y_train, feature_selection)

    if type == 'dt':
        dt = DT(X_train, y_train)
        return predict(dt, DT, X_train, X_test, y_train, feature_selection)

    if type == 'nb':
        nb = NBayes(X_train, y_train)
        return predict(nb, NBayes, X_train, X_test, y_train, feature_selection)

    if type == 'eec':
        eec = EEC(X_train, y_train)
        return predict(eec, EEC, X_train, X_test, y_train, feature_selection)

def predict_with_fa(X_train, X_test, y_train, y_test):
    knn_predict = predict_model('knn', X_train, y_train, X_test, True)
    print("KNN Accuracy")
    print(f1_score(knn_predict, y_test))

    svm_predict = predict_model('svm', X_train, y_train, X_test, True)
    print("SVM Accuracy")
    print(f1_score(svm_predict, y_test))

    rf_predict = predict_model('rf', X_train, y_train, X_test, True)
    print("RF Accuracy")
    print(f1_score(rf_predict, y_test))

    logistic_predict = predict_model('logistic', X_train, y_train, X_test, True)
    print("Logistic Accuracy")
    print(f1_score(logistic_predict, y_test))

    dt_predict = predict_model('dt', X_train, y_train, X_test, True)
    print("DT Accuracy")
    print(f1_score(dt_predict, y_test))

    n_bayes_predict = predict_model('nb', X_train, y_train, X_test, True)
    print("N bayes Accuracy")
    print(f1_score(n_bayes_predict, y_test))

    ecc_predict = predict_model('eec', X_train, y_train, X_test, True)
    print("EEC Accuracy")
    print(f1_score(ecc_predict, y_test))


def predict_without_fa(X_train, X_test, y_train, y_test):
    knn_predict = predict_model('knn', X_train, y_train, X_test)
    print("KNN Accuracy")
    print(f1_score(knn_predict, y_test))

    svm_predict = predict_model('svm', X_train, y_train, X_test)
    print("SVM Accuracy")
    print(f1_score(svm_predict, y_test))

    rf_predict = predict_model('rf', X_train, y_train, X_test)
    print("RF Accuracy")
    print(f1_score(rf_predict, y_test))

    logistic_predict = predict_model('logistic', X_train, y_train, X_test, True)
    print("Logistic Accuracy")
    print(f1_score(logistic_predict, y_test))

    dt_predict = predict_model('dt', X_train, y_train, X_test)
    print("DT Accuracy")
    print(f1_score(dt_predict, y_test))

    n_bayes_predict = predict_model('nb', X_train, y_train, X_test)
    print("N bayes Accuracy")
    print(f1_score(n_bayes_predict, y_test))

    ecc_predict = predict_model('eec', X_train, y_train, X_test)
    print("EEC Accuracy")
    print(f1_score(ecc_predict, y_test))



def perform_cross_validation(X, y):
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    print("Predict using feature selection")
    predict_with_fa(X_train, X_test, y_train, y_test)

    print("Predict using without feature selection")
    predict_without_fa(X_train, X_test, y_train, y_test)

def perform_cross_validation_oversampling(X,y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(X, y)
    perform_cross_validation(X_over, y_over)


def perform_test_report(X, y, X_test):
    X = X[X.columns[:-2]]
    X_test = X_test[X_test.columns[:-2]]
    eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
    eec.fit(X, y)
    dt_predict = eec.predict(X_test)
    np.savetxt('./data/out-final.dat', dt_predict, delimiter=',', fmt='%i')
    print("File save as out-final.dat")

def plot_importance_feature(X, y):
    rf = RF(X, y)
    rf.plot_importance_feature()

    logistic = Logistic(X, y)
    logistic.plot_importance_feature()

    nb = NBayes(X, y)
    nb.plot_importance_feature()

    dt = DT(X, y)
    dt.plot_importance_feature()

    eec = EEC(X, y)
    eec.plot_importance_feature()

    knn = KNN(X, y)
    knn.plot_importance_feature()

    svm = SVM(X, y)
    svm.plot_importance_feature()