#-*- coding:utf -*-

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import nltk
import codecs

def fileLoad(path, b=True):
    with codecs.open(path, 'r', 'utf-8') as f:
        data = np.array(list(map(lambda x: x.split('\t'), f.readlines())))
    
    if b:
        return data[1:, 1], data[1:, 2]
    else:
        return data[1:, 1], data[1:, 0]
# end def

def dataMake(doc, feature):
    aL = np.zeros((len(doc), len(feature)), dtype=np.float32)
    
    for i, doc in enumerate(doc):
        for w in doc:
            if w in feature:
                aL[i][feature[w]] += 1
    
    return np.array(aL)
# end def

if __name__ == '__main__':
    X, y = fileLoad('ratings_test.txt.tw', False)
    
    X = list(map(lambda x: x.split(), X))
    print(X[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test , test_size=0.2, random_state=123)
    
    tokens = [w for row in X_train for w in row]
    text = sorted(nltk.Text(tokens).vocab())
    
    feature = {w:i for i,w in enumerate(text)}
    
    print('train length : %d' %len(X_train))
    print('test length : %d' %len(X_test))
    print('feature length : %d' %len(feature))
    
    X_train = dataMake(X_train, feature)
    X_test = dataMake(X_test, feature)
    
    mL = [LinearSVC(), GaussianNB()]
    for model in mL:
        model.fit(X_train, y_train)
        print(type(LinearSVC()).__name__, classification_report(y_test, model.predict(X_test), target_names=['NEG', 'POS']))