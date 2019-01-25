from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 

ps=PorterStemmer()

example_words=["python","pythoner","pythoning","pythoned"]
# for w in example_words:
# 	print(ps.stem(w))

new_text="it is very important to be pythonly while pythoning with python."
words=word_tokenize(new_text)
for w in words:
 	print(ps.stem(w))