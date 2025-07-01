from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')

text = """Machine learning is fun and powerful.
Deep learning is a branch of machine learning.
I love exploring artificial intelligence topics.
"""

tokenized_sen = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(text)]

model = Word2Vec(tokenized_sen, vector_size=100, window=5, min_count=1, workers = 4)

model.save('word2vec.model')
