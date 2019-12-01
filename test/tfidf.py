# -*- coding: UTF-8 -*-
import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
df = pd.read_csv("data.csv",header=0,encoding='utf-8',dtype = str)
df = df.drop(['Id','过敏史','希望得到的帮助','患病时长','就诊科室','用药情况','链接','既往史','疾病分类'],axis = 1)

stopwords = [line.strip() for line in open('./stopwords/百度停用词表.txt',encoding='UTF-8').readlines()]
jieba.analyse.set_stop_words('./stopwords/哈工大停用词表.txt') 
jieba.analyse.set_idf_path('./idf_file.txt')

#tfidf_file = open('tfidf_file_with_default_idf.txt','w')
tfidf_file = open('tfidf_file.txt','w')
for index,row in df.iterrows():
    row_content = ""
    
    for col in row:
        col= re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", str(col))
        row_content += col
    
    tags = jieba.analyse.extract_tags(row_content, topK=10 ,withWeight=True, allowPOS=('nv','n','vn'))
    for tag in tags:
        #tfidf_file.writelines(tag[0]+':'+str(tag[1])+' ') 带weight输出
        tfidf_file.writelines(tag[0]+' ') 
    tfidf_file.write('\n')    


    
