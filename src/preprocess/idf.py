# -*- coding: UTF-8 -*-
import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
from collections import Counter

df = pd.read_csv("./data/raw_data.csv",header=0,encoding='utf-8',dtype = str)
df = df.drop(['Id','过敏史','希望得到的帮助','患病时长','就诊科室','用药情况','链接','既往史'],axis = 1)
stopwords = [line.strip() for line in open('./stopwords/中文停用词表.txt',encoding='UTF-8').readlines()]
#jieba.analyse.set_stop_words('./stopwords/中文停用词表.txt') 

freq_list = Counter()
for index,row in df.iterrows():
    words = []
    repeat = []
    row_content = ""
    
    for col in row:
        col = re.sub(r"[0-9a-zA-Z\s+\.\!\/_,$%^*(+\"\']+|[+-——！`；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+", " ", str(col))
        row_content += col
    
    words = jieba.cut(row_content,cut_all = False)
    for word in words:
        if word != ' ' and word not in stopwords and word not in repeat:
            freq_list[word] +=1 
            repeat.append(word)


idf_file = open("./data/idf_file.txt",'w')
for word in freq_list.keys():
    cnt = int(freq_list.get(word))
    idf = np.log10(df.shape[0]/(1.0*cnt))
    idf_file.writelines(word+' '+str(idf)+'\n')


    
