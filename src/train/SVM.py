import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,roc_curve,auc)
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import sys 
sys.path.append('./src') 
from evaluate.auc import aucPlot
from evaluate.multi_auc import multiAucPlot 

n_class = 5
vec_dimension = 200
df = pd.read_csv('./data/dataset_phrase_with_stopwords.csv',header = 0)
#df = pd.read_csv('./data/dataset_idf.csv',header = 0)#准确率更低了。。。
data = df.values
x = data[:,1:vec_dimension]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=50))
        #("svm_clf",SVC(kernel = "rbf",gamma = 0.01, C=5))
        #("svm_clf", SVC(kernel = "sigmoid" ,gamma = 0.01,coef0= 0, C=5))
    ])
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_score = clf.decision_function(x_test)
y_one_hot = label_binarize(y_test, np.arange(n_class))
print("accuracy_score:")
print(accuracy_score(y_test, y_pred, normalize=True))
print(classification_report(y_test, y_pred, labels=None,
                            target_names=None, sample_weight=None, digits=2))
print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))


#aucPlot(y_one_hot,y_score) 
multiAucPlot(y_one_hot,y_score,n_class)
