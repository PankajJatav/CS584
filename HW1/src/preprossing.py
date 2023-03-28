import numpy as np
import nltk
import pandas as pd

from util import cleanhtml, lemmatize_sentence
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, words
from nltk.tokenize import WhitespaceTokenizer

tk = WhitespaceTokenizer()
words = set(words.words())

class Prepossing:
    def __init__(self, file_loc, file_name, no_of_col):
        self.analyzer = TfidfVectorizer().build_analyzer()
        self.stemmer = SnowballStemmer(language="english", ignore_stopwords=True)
        with open(file_loc, encoding='UTF-8') as fp:
            data = fp.readlines()
            self.data = data

        df = pd.DataFrame([x.split('\t') for x in self.data])
        df = df.drop(list(range(no_of_col, df.shape[1])), axis=1)
        if no_of_col == 1:
            self.df = df.set_axis(['Review'], axis=1, inplace=False)
        elif no_of_col == 2:
            self.df = df.set_axis(['Rating', 'Review'], axis=1, inplace=False)
        print(1)
        self.df['Processed'] = self.df['Review'].apply(self.stemmed_doc)
        print(2)
        self.df.to_csv('./data/'+file_name+'.csv', index=False)

    def stemmed_doc(self, doc):
        doc = cleanhtml(doc)
        tk_data = tk.tokenize(doc)
        list_data = [w for w in tk_data if w.lower() not in stopwords.words('english')]
        list_data = [w.lower() for w in list_data if w.lower() in words and w.isalpha()]
        lem_data = lemmatize_sentence(list_data)
        data = [self.stemmer.stem(w) for w in self.analyzer(lem_data)]
        return data
