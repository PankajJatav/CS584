import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk.tokenize as tok
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

lemmatizer = WordNetLemmatizer()
tokr = tok.toktok.ToktokTokenizer()

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(sentence)
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def select_model_selection():
    model_selection = 0
    while (1):
        print("Please choose the model selection")
        print("1. 66 Training Set - 34 Test Set")
        print("2. 70 Training Set - 30 Test Set")
        print("3. 80 Training Set - 20 Test Set")
        print("4. 90 Training Set - 10 Test Set")
        choice = input("Enter your choice: ")
        if not choice.isnumeric():
            print("Oops, Please try again")
            continue
        choice = int(choice)
        if choice == 1:
            model_selection = 0.66
            break
        elif choice == 2:
            model_selection = 0.7
            break
        elif choice == 3:
            model_selection = 0.8
            break
        elif choice == 4:
            model_selection = 0.9
            break
        else:
            print("Oops, Please try again")
    return model_selection

def top_features (vec):
    indices = np.argsort(vec.idf_)[::-1]
    print(vec.idf_)
    features = vec.get_feature_names()

    top_features = [features[i] for i in indices[:50]]
    print(top_features)

def perform_vectorization(train_data, test_data):
    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit(train_data)
    train_vec = vec.transform(train_data)
    test_vec = vec.transform(test_data)
    # top_features(vec) #uncommnet to see top features
    return train_vec, test_vec

# This function will return the K value with higher accuracy
def perform_knn_values(training_vec, test_vec, train_class, test_class):
    test_vec = test_vec.transpose()
    value = np.matmul(training_vec.toarray(), test_vec.toarray())

    def find_knn(row):

        score = 0
        arr = []
        for k in range(1, 250, 2):
            for w in row.argsort()[-k:][::-1]:
                score += train_class[w]
            if (score > 0):
                arr.append(1)
            else:
                arr.append(-1)

    predict = np.apply_along_axis(find_knn, axis=0, arr=value)
    k_value = 0
    acc = 0
    index = 0
    for data in predict.transpose():
        accuracy = accuracy_score(data, test_class)
        if(acc < accuracy):
            k_value = index
            acc = accuracy
        index += 1
    return k_value



def perform_knn(training_vec, test_vec, train_class):
    def find_knn(row):
        score = 0
        for w in row.argsort()[-221:][::-1]:
            score += train_class[w]
        if (score > 0):
            return 1
        else:
            return -1

    test_vec = test_vec.transpose()
    value = np.matmul(training_vec.toarray(), test_vec.toarray())
    # save_plot(value, 'knn.png', False, False) #uncommnet to plot graph
    predict = np.apply_along_axis(find_knn, axis=0, arr=value)
    return predict

def print_result(prediction, train_class):
    print(confusion_matrix(train_class, prediction))
    print(accuracy_score(train_class, prediction))

def save_plot(mat, file_name , label=True, to_array = True):

    if (to_array):
        plot_data = mat.toarray()
    else:
        plot_data = mat
    plot_data[plot_data == 0] = 'nan'

    plt.matshow(plot_data)
    plt.colorbar()
    plt.xticks(rotation=90)
    if label:
        plt.xlabel('features')
        plt.ylabel('data')
    plt.savefig('./data/' + file_name, dpi=200)
    return 0

def corss_validation():
    df_train = pd.read_csv('./data/processed_train_data.csv')
    model_selection = select_model_selection()
    train_data, test_data = np.array_split(df_train['Processed'], [int(model_selection * len(df_train['Processed']))])
    training_class, test_class = np.array_split(df_train['Rating'], [int(model_selection * len(df_train['Rating']))])

    print("Performing vectorization")
    train_vec, test_vec = perform_vectorization(train_data, test_data)
    print("Completed vectorization")

    # save_plot(train_vec, 'train_cross.png') #uncommnet to plot graph
    # save_plot(test_vec, 'test_cross.png') #uncommnet to plot graph

    print("Performing KNN")
    predict = perform_knn(train_vec, test_vec, training_class)
    print("Completed KNN")
    print_result(test_class, predict)

def test_prediction():
    df_train = pd.read_csv('./data/processed_train_data.csv')
    df_test = pd.read_csv('./data/processed_test_data.csv')
    train_data = df_train['Processed']
    test_data = df_test['Processed']
    train_class = df_train['Rating']

    print("Performing vectorization")
    train_vec, test_vec = perform_vectorization(train_data, test_data)

    # save_plot(train_vec, 'train_predict.png') #uncommnet to plot graph
    # save_plot(test_vec, 'test_predict.png') #uncommnet to plot graph

    print("Completed vectorization")
    print("Performing KNN")
    predict = perform_knn(train_vec, test_vec, train_class)
    print("Completed KNN")

    np.savetxt('./data/out.dat', predict, delimiter=',', fmt='%i')

    print("Output can be find at './data/out.dat' file")