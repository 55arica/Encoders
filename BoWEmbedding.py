from sklearn.feature_extraction.text import CountVectorizer

texts = ["Currently, i am watching wind breaker season 2"]

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(texts) # CountVectorizer stores the learned information (vocabulary), and it reuses it for future inputs.
print(vectorizer.vocabulary_)
print(x.toarray())

