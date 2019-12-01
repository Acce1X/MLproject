# -*- coding: UTF-8 -*-
import pandas as pd 
import jieba 
import jieba.analyse
import numpy as np
import re
from collections import Counter

'''
fq0,该词正文本出现频率
fq1,该词负文本出现频率
fq2,该词正文本不出现频率
fq3,该词负文本不出现频率

'''

'''
{
    'word':[cnt_p,cnt_n]
}
'''
def stat(df,tag):
    fq = {}
    sum_p = 0
    sum_n = 0
    for i,row in df.iterrows():
        row_text = ""
        repeat = []

        if(row['tag'] == tag):
            sum_p +=1 
        else: 
            sum_n += 1 
        
        for col in row:
            col = re.sub(r"[0-9a-zA-Z\s+\.\!\/_,$%^*(+\"\']+|[+-——！；「」》:：“”`·‘’《，。？、~@#￥%……&*（）()]+", " ", str(col))
            row_text+=col

        words = jieba.cut(row_text,cut_all = False)
        
        for word in words:
            if word not in stopwords:
                if(fq.get(word) == None):
                    fq.update({word:[0,0,0,0]})
                if word != ' ' and word not in repeat:
                    if row['tag'] == tag :
                        fq[word][0] += 1 
                        repeat.append(word)
                    else: 
                        fq[word][1] += 1 
                        repeat.append(word)
    return  fq,sum_p,sum_n



def chi2(fq,sum_p,sum_n):
    chi = {}
    for word in fq.keys():
        a = fq[word][0] / sum_p 
        b = fq[word][1] / sum_n
        fq[word][2] = sum_p - a
        fq[word][3] = sum_n - b
        c = fq[word][2] / sum_p 
        d = fq[word][3] / sum_n
        n = sum_p + sum_n
        val = float(n*(a*d - b*c)*(a*d - b*c) / (a+c)*(a+b)*(b+d)*(b*c))
        chi.update({word:val})
    result = sorted(chi,key = lambda x:chi[x],reverse=True)
    return result,chi
    
df = pd.read_csv("data.csv",header=0,encoding='utf-8',dtype = str)
df = df.drop(['Id','过敏史','希望得到的帮助','患病时长','就诊科室','用药情况','链接','既往史'],axis = 1)
stopwords = [line.strip() for line in open('./stopwords/中文停用词表.txt',encoding='UTF-8').readlines()]
illness_tag = {'胃炎':1,'胃癌':2,'肠胃炎':3,'慢性前变性胃炎':4,'慢性胃炎':5}
tag2ill = ['','胃炎','胃癌','肠胃炎','慢性前变性胃炎','慢性胃炎']        
df['tag'] = df.apply(lambda row:illness_tag[row['疾病分类']],axis = 1)
df.drop(['疾病分类'],axis = 1)

for i in range(1,6):
    fq,sum_p,sum_n = stat(df,i)
    result,chi = chi2(fq,sum_p,sum_n)
    chifile = open('./chi_result/chifile_'+str(i)+'.txt','w')    
    for key in result:
        chifile.write(key+':'+str(chi[key])+'\n')
    chifile.close()







