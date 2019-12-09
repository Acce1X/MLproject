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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import datapath


vec_dimension = 200
txt_path = '/Users/lichuanhan/Codes/MLProject/data/word_with_stopwords.txt'
sentences = LineSentence(datapath(txt_path))
phrases = Phrases.load('./data/phraser.pkl')
df = pd.read_csv('./data/word.csv',header=0,encoding='utf-8',dtype = str)
doc_vecs = np.zeros((1,vec_dimension))
documents = [TaggedDocument(phrases[doc], [i]) for i, doc in enumerate(sentences)]
model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=4)

df1 = pd.DataFrame(model.docvecs.vectors_docs)
dataset = pd.concat([df1,df['tag']],axis = 1)
dataset.to_csv('./data/dataset_doc2vec.csv')


