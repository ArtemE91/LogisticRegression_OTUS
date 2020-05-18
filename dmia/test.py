import pandas as pd
import numpy as np
import tqdm  # interactive progress bar
from sklearn.feature_extraction.text import TfidfVectorizer
from dmia.gradient_check import *
from dmia.classifiers.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train_df = pd.read_csv('../data/train.csv')

review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]

# Приводим строки к векторному виду
vectorizer = TfidfVectorizer()
tfidfed = vectorizer.fit_transform(review_summaries)

X = tfidfed  # N x D массив данных
y = train_df.Prediction.values  # одномеррный массив для 2 классов (0, 1)
# разбиваем массива на случайный набор данных где:
# train_size - доля набора данных для включения
# random_state - управляет перетасовкой
# X_train, y_train = 70%, X_test, y_test = 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_train_sample = X_train[:10000]
y_train_sample = y_train[:10000]
clf = LogisticRegression()

# Получаем массив длиной N + 1 случайных значений с нормальным распределением
clf.w = np.random.randn(X_train_sample.shape[1] + 1) * 2
clf.train(X_train, y_train, learning_rate=0.1, num_iters=1000, batch_size=1000, reg=0.1, verbose=True)

train_scores = []
test_scores = []
num_iters = 1000

for i in tqdm.trange(num_iters):
    # Сделайте один шаг градиентного спуска с помощью num_iters=1
    clf.train(X_train, y_train, learning_rate=1.0, num_iters=1, batch_size=256, reg=1e-3)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(10, 8))
plt.plot(train_scores, 'r', test_scores, 'b')
plt.show()

pos_features = np.argsort(clf.w)[-10:]
neg_features = np.argsort(clf.w)[:10]

fnames = vectorizer.get_feature_names()
print([fnames[p] for p in pos_features])
print([fnames[n] for n in neg_features])



