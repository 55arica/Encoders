from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "i have done watching wind breaker season 2 anime",
    "and good thing is i have already watched the first season of this anime"
]

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(texts)

print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out())
print(x.toarray)
