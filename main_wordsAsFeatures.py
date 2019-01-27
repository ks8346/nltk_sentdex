import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers=classifiers
	
	def classify(self,features):
		votes=[]
		for c in self._classifiers:
			v=c.classify(features)
			votes.append(v)
		return mode(votes)	
	
	def confidence(self,features):
		votes=[]
		for c in self._classifiers:
			v=c.classify(features)
			votes.append(v)	
		choice_votes=votes.count(mode(votes))
		conf=choice_votes/len(votes)
		return conf					
						




documents=[(list(movie_reviews.words(fileid)), category)
		for category in movie_reviews.categories()
		for fileid in movie_reviews.fileids(category)]

# documents=[]

# for category in movie_reviews.categories():
# 	for fileid in movie_reviews.fileids(category):
# 		documents.append(list(movie_reviews.words(fileid)))		
# random.shuffle(documents)

#print(documents[1])

all_words=[]
for w in movie_reviews.words():
	all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)	
word_features=list(all_words.keys())[:3000]

def find_features(document):
	words=set(document)
	features={}
	for w in word_features:
		features[w]=(w in words)
	return features
		
#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets=[(find_features(rev),category) for (rev,category) in documents]
#positive
training_set=featuresets[:1900]
testing_set=featuresets[1900:]

#negative
training_set=featuresets[100:]
testing_set=featuresets[:100]
#classifier=nltk.NaiveBayesClassifier.train(training_set)

classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes algo accuracy:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Naive Bayes algo accuracy:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

# G_classifier=SklearnClassifier(GaussianNB())
# G_classifier.train(training_set)
# print("G Naive Bayes algo accuracy:",(nltk.classify.accuracy(G_classifier,testing_set))*100)

LR_classifier=SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print("LR accuracy:",(nltk.classify.accuracy(LR_classifier,testing_set))*100)

SGD_classifier=SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD accuracy:",(nltk.classify.accuracy(SGD_classifier,testing_set))*100)

SVC_classifier=SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy:",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)

LSVC_classifier=SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
print("LSVC accuracy:",(nltk.classify.accuracy(LSVC_classifier,testing_set))*100)

NSVC_classifier=SklearnClassifier(NuSVC())
NSVC_classifier.train(training_set)
print("NSVC accuracy:",(nltk.classify.accuracy(NSVC_classifier,testing_set))*100)

B_classifier=SklearnClassifier(BernoulliNB())
B_classifier.train(training_set)
print("B Naive Bayes algo accuracy:",(nltk.classify.accuracy(B_classifier,testing_set))*100)

voted_classifier=VoteClassifier(classifier,
	MNB_classifier,
	B_classifier,
	LR_classifier,
	SGD_classifier,
	LSVC_classifier,
	NSVC_classifier)
print("voted_classifier algo accuracy:",(nltk.classify.accuracy(voted_classifier,testing_set))*100)


# print("Classification:",voted_classifier.classify(testing_set[1][0]),"Confidence:",voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[2][0]),"Confidence:",voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[3][0]),"Confidence:",voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:",voted_classifier.classify(testing_set[4][0]),"Confidence:",voted_classifier.confidence(testing_set[4][0])*100) 



