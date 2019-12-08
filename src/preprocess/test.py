import pandas as pd 
from gensim.models import word2vec 
model = word2vec.Word2Vec.load('./data/vec.model')
print(model.wv['胃炎'])