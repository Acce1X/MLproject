import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/dataset_phrase.csv',header = 0)
#df = pd.read_csv('./dataset_idf.csv',header = 0)#准确率更低了。。。
data = df.values
x = data[:,1:100]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

clf = MLPClassifier(solver='lbfgs',activation='logistic',hidden_layer_sizes=(25,),alpha=1e-5)  # 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred,normalize=True))
print(metrics.classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
print(metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))