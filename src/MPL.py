import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./dataset.csv',header = 0)
#df = pd.read_csv('./dataset_idf.csv',header = 0)#准确率更低了。。。
data = df.values
x = data[:,1:100]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100,5), random_state=1)  # 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred,normalize=True))

