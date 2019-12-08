import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
import random
from gensim.models import word2vec 
from sklearn.manifold import TSNE 

model = word2vec.Word2Vec.load('./data/vec.model')

#df = pd.read_csv('./test.csv',header=0,encoding='utf-8',dtype = str)

vocab_list =[]


for i in range(0,200):
    v = list(model.wv.vocab.keys())
    index = random.randint(0,len(v))
    vocab_list.append(v[index])
    
X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(model.wv[vocab_list])

from matplotlib.font_manager import *  
import matplotlib.pyplot as plt 
#解决负号'-'显示为方块的问题  
plt.figure(figsize=(14, 8)) 
myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

plt.scatter(X_tsne[:,0],X_tsne[:,1])
for i in range(len(X_tsne)):
    x=X_tsne[i][0]
    y=X_tsne[i][1]
    plt.text(x , y ,vocab_list[i], fontproperties= myfont,size = 8)
 
plt.show()