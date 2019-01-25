from nltk.corpus import wordnet

syns=wordnet.synsets("program")
# #synset
# print(syns[0].name())
# #just word
# print(syns[0].lemmas()[0].name())
# #definition
# print(syns[0].definition())
# #examples
print(syns[0].examples())


synonyms=[]
antonyms=[]

for syns in wordnet.synsets("good"):
	for l in syns.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print("/////")
print(set(antonyms))

w1=wordnet.synset("ship.n.01")

w2=wordnet.synset("dog.n.01")

w1=wordnet.synset("kid.n.01")

w2=wordnet.synset("kitten.n.01")

#similarity

print(w1.wup_similarity(w2))

