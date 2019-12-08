# -*- coding: UTF-8 -*-
import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
from gensim.models import word2vec 
import time

'''
分词预处理，疾病标记tag，存储为word.csv
'''
'''
wordvec_set = set()
for i in range(1,6):
    chifile = open('./chi_result/chifile_'+str(i)+'.txt','r')    
    for j in range(0,1000):
        line = chifile.readline()
        wordvec_set.add(line.split(':')[0])
    chifile.close()

print(len(wordvec_set))
'''

df = pd.read_csv("./data/raw_data.csv",header=0,encoding='utf-8',dtype = str)
df = df.drop(['Id','过敏史','希望得到的帮助','患病时长','就诊科室','用药情况','链接','既往史'],axis = 1)
stopwords = [line.strip() for line in open('./stopwords/中文停用词表.txt',encoding='UTF-8').readlines()]
illness_tag = {'胃炎':1,'胃癌':2,'肠胃炎':3,'慢性前变性胃炎':4,'慢性胃炎':5}
tag2ill = ['','胃炎','胃癌','肠胃炎','慢性前变性胃炎','慢性胃炎']        
df['tag'] = df.apply(lambda row:illness_tag[row['疾病分类']],axis = 1)
df.drop(['疾病分类'],axis = 1)


word_frame = []
word_set = set()

word_file = open('./data/word.txt','w')

for index,row in df.iterrows():
    row_text = ""
    for col in row:
        col = re.sub(r"[0-9a-zA-Z\s+\.\!\/_,$%^*(+\"\']+|[+-——！`；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+", " ", str(col))
        row_text+=col

    words = jieba.cut(row_text,cut_all = False)
    valid_words = []
    
    for word in words:
        if word not in stopwords:
            valid_words.append(word)        
    word_file.write(' '.join(valid_words)+'\n')
    #word_frame.append(' '.join(valid_words))

word_file.close()
'''
tag_frame = df['tag']
frame_dict = {'text':word_frame,'tag':tag_frame}
new_df = pd.DataFrame(frame_dict)
new_df.to_csv('./data/word.csv')
'''

