
import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
from gensim.models import word2vec 
import time
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import datapath

stopwords = [line.strip() for line in open('./stopwords/中文停用词表.txt',encoding='UTF-8').readlines()]
txt_path = '/Users/lichuanhan/Codes/MLProject/data/word_with_stopwords.txt'
sentences = LineSentence(datapath(txt_path))
phrases = Phrases(sentences,threshold=4.8,common_terms=stopwords)
model = word2vec.Word2Vec(phrases[sentences],size = 200,window = 3)

model.save('./data/vec_phrase_with_stopwords.model')
y2 = model.most_similar(u"慢性_胃炎", topn=20) # 20个最相关的
print (u"和【胃炎】最相关的词有：\n")
for item in y2:
  print (str(item[0])+str(item[1]))
print ("--------\n")

