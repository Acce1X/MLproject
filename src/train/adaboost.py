import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


vec_dimension = 200
df = pd.read_csv('./data/dataset_phrase_with_stopwords.csv',header = 0)
#df = pd.read_csv('./dataset_idf.csv',header = 0)#准确率更低了。。。
data = df.values
x = data[:,1:200]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_split=20,),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.7)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred,normalize=True))
print(metrics.classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
print(metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))