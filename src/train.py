
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./dataset.csv')
data = df.values
x = data[:,1:99]
y = data[:,-1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])

poly100_kernel_svm_clf.fit(X=x_train, y=y_train)  # 训练模型。参数sample_weight为每个样本设置权重。应对非均衡问题
y_pred = poly100_kernel_svm_clf.predict(x_test)  # 使用模型预测值
print(accuracy_score(y_test, y_pred,normalize=True))

