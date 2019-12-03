import pandas as pd 
from gensim.models import word2vec 
model = word2vec.Word2Vec.load('./vec.model')
vec = model.wv['胃炎']
print(vec)