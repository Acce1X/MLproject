import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
import random
from gensim.models import word2vec 
from sklearn.preprocessing import normalize 

from sklearn.manifold import TSNE 

model = word2vec.Word2Vec.load('./vec.model')

df = pd.read_csv('./word.csv',header=0,encoding='utf-8',dtype = str)


vecs = np.zeros((1,100))

for index,row in df.iterrows():
    words = row[1].split(' ')
    vec = np.zeros((1,100))
    cnt = 0
    for word in words:
        if word !='' and word in model.wv.vocab.keys():
            vec = vec + model.wv[word]
            cnt += 1
    vec = vec / cnt
    vec = normalize(vec,axis = 1,norm='max')
    vecs = np.append(vecs,vec,axis = 0 )

vecs = np.delete(vecs,0,axis = 0)

df1 = pd.DataFrame(vecs)
dataset = pd.concat([df1,df['tag']],axis = 1)
dataset.to_csv('./dataset.csv')


