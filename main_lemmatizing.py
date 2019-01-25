from nltk.stem import WordNetLemmatizer
lematizer=WordNetLemmatizer()
print(lematizer.lemmatize("better",pos="a"))
print(lematizer.lemmatize("best",pos="a"))
print(lematizer.lemmatize("run",pos="v"))