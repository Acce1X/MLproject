from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import matplotlib.pyplot as plt
import pandas as pd


vec_dimension = 200
df = pd.read_csv('./data/dataset_phrase_with_stopwords.csv',header = 0)
#df = pd.read_csv('./dataset_idf.csv',header = 0)
data = df.values
X = data[:,1:vec_dimension]
y = data[:,-1].astype(int)
cValue = ['r','y','g','b','r','y','g','b','r']  

X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
X_tsne = TSNE(n_components=2, init='random', random_state=0).fit_transform(X)
X_pca = PCA().fit_transform(X)

def plot_scatter(X,y,label,sub_no):
    plt.subplot(sub_no)
    tag2ill = ['胃炎','胃癌','肠胃炎','慢性前变性胃炎','慢性胃炎'] 
    for i in range(0,5):
        px = X[:, 0][y == i]
        py = X[:, 1][y == i]                                                       
        plt.scatter(px, py,label = label)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
    #'plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(tag2ill)


plot_scatter(X_tsne,y,'X_tsne',131)
plot_scatter(X_pca,y,'X_pca',132)
plot_scatter(X_lda,y,'X_lda',133)
plt.show()
