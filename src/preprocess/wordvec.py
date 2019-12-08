import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
from gensim.models import word2vec 
import time
word_set = set()

df = pd.read_csv('./data/word.csv',header=0,encoding='utf-8',dtype = str)
sentences = []
for index,row in df.iterrows():
    sentence =[]
    text = row[1].split(' ')
    for word in text:
        if word !='':
            sentence.append(word)
            word_set.add(word)
    sentences.append(sentence)

start = time.clock()
model = word2vec.Word2Vec(sentences,size = 200,window = 1)
model.save('./data/vec.model')
end = time.clock()

print('Running time: %s Seconds'%(end-start))


y2 = model.most_similar(u"胃炎", topn=20) # 20个最相关的
print (u"和【胃炎】最相关的词有：\n")
for item in y2:
  print (str(item[0])+str(item[1]))
print ("--------\n")
