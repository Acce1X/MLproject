# -*- coding: UTF-8 -*-
import pandas as pd 
import jieba 
import jieba.analyse
from collections import Counter
df = pd.read_csv("data.csv",header=0,encoding='utf-8',dtype = str)

df = df.drop(['Id','过敏史','希望得到的帮助','患病时长','就诊科室','用药情况','链接','既往史','疾病分类'],axis = 1)
df.head(5)

test = df.head(5)

stopwords = [line.strip() for line in open('./stopwords/哈工大停用词表.txt',encoding='UTF-8').readlines()]

content =[]
freq_list = Counter()
for index,row in df.iterrows():
    words = []
    for col in row:
        words = jieba.cut(str(col),cut_all = False)
        content += jieba.cut(str(col),cut_all = False)

        for word in words:
            if word not in stopwords:
                freq_list[word] +=1 

outfile = open('words_freq.txt','w')
for word in freq_list.keys():
    cnt = freq_list.get(word)
    if(cnt>10):
        outfile.writelines(word+':'+str(cnt)+'\n')

    #print("Default Mode: " + "/ ".join(seg_list))


jieba.analyse.set_stop_words('./stopwords/哈工大停用词表.txt') 
tags = jieba.analyse.extract_tags(str(content), topK=80, allowPOS=('ns', 'n', 'vn', 'v'))

outfile = open('keywords.txt','w')
for tag in tags:
    outfile.writelines(str(tag)+'\n')

outfile = open('keycol.txt','w')
for index,row in df.iterrows():
    words = []
    for col in row:
        words = jieba.cut(str(col),cut_all = False)
        for word in words:
            if word in tags:
                outfile.write(str(word)+' ')
        outfile.write('\n')
