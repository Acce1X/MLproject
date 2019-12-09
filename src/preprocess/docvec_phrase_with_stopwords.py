import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
import random
from gensim.models import word2vec 
from sklearn.preprocessing import normalize 
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import datapath


from sklearn.manifold import TSNE 

vec_dimension = 200
model = word2vec.Word2Vec.load('./data/vec_phrase_with_stopwords.model')

import os
txt_path = os.getcwd() +'/data/word_with_stopwords.txt'
sentences = LineSentence(datapath(txt_path))

stopwords = [line.strip() for line in open('./stopwords/中文停用词表.txt',encoding='UTF-8').readlines()]
phrases = Phrases.load('./data/phraser.pkl')
df = pd.read_csv('./data/tag.csv',header=0,encoding='utf-8',dtype = str)

doc_vecs = np.zeros((1,vec_dimension))
for sentence in sentences:
    sentence = phrases[sentence]
    doc_vec = np.zeros((1,vec_dimension))
    for word in sentence:
        if word in model.wv.vocab.keys() and word not in stopwords:
            doc_vec += model.wv[word]
    doc_vec /= len(sentence)
    doc_vec = normalize(doc_vec,axis = 1,norm='max')
    doc_vecs = np.append(doc_vecs,doc_vec,axis = 0)
doc_vecs = np.delete(doc_vecs,0,axis = 0)

df1 = pd.DataFrame(doc_vecs)
dataset = pd.concat([df1,df['tag']],axis = 1)
dataset.to_csv('./data/dataset_phrase_with_stopwords.csv')


