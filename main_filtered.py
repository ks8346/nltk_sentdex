from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

example_text="hello there, how are you doing today? The weather is great and Python is awesome. The sky is blue."

stop_words=set(stopwords.words("english"))
words=word_tokenize(example_text)
# filtered_sent=[]

# for w in words:
# 	if w not in stop_words:
# 		filtered_sent.append(w)
# print(filtered_sent)

filtered_sent=[w for w in words if not w in stop_words]
print(filtered_sent)