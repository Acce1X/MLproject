from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import datapath

txt_path = '/Users/lichuanhan/Codes/MLProject/data/word.txt'
sentences = LineSentence(datapath(txt_path))
phrases = Phrases(sentences,threshold = 4.8)
phraser = Phraser(phrases)
phraser.save('./data/phraser.pkl')

