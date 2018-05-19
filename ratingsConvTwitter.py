#-*- coding:utf-8 -*-

from konlpy.tag import Twitter
import numpy as np
import codecs
import progressbar

t = Twitter()

with codecs.open('ratings_test.txt', 'r', 'utf-8') as f:
    data = np.array([row.split('\t') for row in f.readlines()[1:]])

X, y = data[:, 1], data[:, 2].astype(np.int32)

with codecs.open('ratings_test.txt.tw', 'w', 'utf-8') as f:
    bar = progressbar.ProgressBar()
    for doc, label in bar(zip(X, y)):
        doc = ' '.join(['/'.join(row) for row in t.pos(doc, norm=True, stem=True)])
        f.write(str(label) + '\t' + doc + '\n')