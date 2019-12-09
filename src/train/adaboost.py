import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score)
from sklearn.preprocessing import label_binarize

from sklearn.ensemble import RandomForestClassifier

import sys 
sys.path.append('./src') 
from evaluate.auc import aucPlot 
from evaluate.multi_auc import multiAucPlot 

n_class = 5
vec_dimension = 200
df = pd.read_csv('./data/dataset_phrase_with_stopwords.csv',header = 0)
#df = pd.read_csv('./dataset_idf.csv',header = 0)#准确率更低了。。。
data = df.values
x = data[:,1:200]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',min_samples_split=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.7)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred = clf.predict(x_test)
y_score = clf.predict_proba(x_test)
y_one_hot = label_binarize(y_test, np.arange(n_class))
print("accuracy_score:")
print(accuracy_score(y_test, y_pred,normalize=True))
print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))


#aucPlot(y_one_hot,y_score) 
multiAucPlot(y_one_hot,y_score,n_class)