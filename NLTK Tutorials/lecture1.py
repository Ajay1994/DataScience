# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:04:40 2016

@author: Pokemon
"""

import nltk
nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
#Corpus - Body of text, singular. Corpora is the plural of this. Example: A collection of medical journals.

#Lexicon - Words and their meanings. Example: English dictionary. Consider, however, 
#that various fields will have different lexicons. For example: To a financial investor, the first meaning
#for the word "Bull" is someone who is confident about the market, as compared to the common English lexicon, 
#where the first meaning for the word "Bull" is an animal. As such, there is a special lexicon for financial 
#investors, doctors, children, mechanics, and so on.

#Token - Each "entity" that is a part of whatever was split up based on rules. For examples,
# each word is a token when a sentence is "tokenized" into words. Each sentence can also be a token,
# if you tokenized the sentences out of a paragraph.


#########################################################################
######################### Tokenization ##################################
#########################################################################
EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))

print(word_tokenize(EXAMPLE_TEXT))

for i in word_tokenize(EXAMPLE_TEXT):
    print i

##########################################################################
########################### Stop Words ###################################
##########################################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This country is worth living. I am having a very good time here in hongkong"

stop_words = set(stopwords.words("english"))

# print set of stop words
print(stop_words)

words = word_tokenize(example_sent)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

filtered_sentence = [w for w in words if not w in stop_words]


##########################################################################
############################# Stemming ###################################
##########################################################################

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))
    
new_text =  "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))


##########################################################################
############################# POS Tagging ################################
##########################################################################

"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when

"""

from nltk.corpus import state_union
#PunktSentenceTokenizer. This tokenizer is capable of unsupervised machine 
#learning, so you can actually train it on any body of text that you use.
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

print(train_text)

#Next, we can train the Punkt tokenizer like:
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

#Then we can actually tokenize, using:
tokenized = custom_sent_tokenizer.tokenize(sample_text)

#print the tokenized sentences
print(tokenized)

def process_content():
    try:
        for sent in tokenized:
            words = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(words)
            print(tagged)
            
    except Exception as e:
        print(str(e))

process_content()

##########################################################################
############################# Chucnking  ################################
##########################################################################
"""
One of the main goals of chunking is to group into what are known as "noun phrases." 
These are phrases of one or more words that contain a noun, maybe some descriptive words, 
maybe a verb, and maybe something like an adverb. The idea is to group nouns with 
the words that are in relation to them.

what is happening here is our "chunked" variable is an NLTK tree. Each "chunk" and "non chunk" is a "subtree"
of the tree. We can reference these by doing something like chunked.subtrees. We can then iterate through these 
subtrees like so:

        for subtree in chunked.subtrees():
                print(subtree)
                
    
"""

def process_content():
    try:
        for i in tokenized[2:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()

"""
Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk. 
The chunk that you remove from your chunk is your chink.
The code is very similar, you just denote the chink, after the chunk, with }{ instead of the chunk's {}.

Now, the main difference here is:

}<VB.?|IN|DT|TO>+{
This means we're removing from the chink one or more verbs, prepositions, determiners, or the word 'to'.
"""

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()


##############################################################################
######################## NAMED Entuty Recognition ############################
##############################################################################
"""
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian

Here, with the option of binary = True, this means either something is a named entity, or not. 


When Binary is False, it picked up the same things, but wound up splitting up terms like 
White House into "White" and "House" as if they were different, whereas we could see in the binary = True option, 
the named entity recognition was correct to say White House was part of the same named entity.
"""


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()


##############################################################################
################################ Lemmetization ###############################
##############################################################################

"""
A very similar operation to stemming is called lemmatizing. The major difference between these is, 
as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary,
but you can look up a lemma.Some times you will wind up with a very similar word, but sometimes, 
you will wind up with a completely different word.

"""

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a")) #Parts of speech Adjective
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))

##############################################################################
#################################### Corpora  ################################
##############################################################################
"""
Depending on your installation, your nltk_data directory might be hiding in a multitude of locations.
To figure out where it is, head to your Python directory, where the NLTK module is.

import nltk
print(nltk.__file__)

"""

import nltk
print(nltk.__file__)

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text is loaded from corpora
sample = gutenberg.raw("bible-kjv.txt")

#Tokenize the corpora
tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])



##############################################################################
#################################### Wordnet #################################
##############################################################################

#Use WordNet alongside the NLTK module to find the meanings of words, synonyms, 
#antonyms, and more.

from nltk.corpus import wordnet

#Print sysnsets of program
syns = wordnet.synsets("program")

print(syns)
"""
[Synset('plan.n.01'), Synset('program.n.02'), Synset('broadcast.n.02'), 
Synset('platform.n.02'), Synset('program.n.05'), Synset('course_of_study.n.01'), 
Synset('program.n.07'), Synset('program.n.08'), Synset('program.v.01'), 
Synset('program.v.02')]
"""

print(syns[0].name())

print(syns[0].lemmas()[0].name())

#Definition of that first synset:
print(syns[0].definition())

#Examples of the word in use:
print(syns[0].examples())

#getting synonyms and antonyms of a word.

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())



print(set(synonyms))
print(set(antonyms))

#######################################################
# Comparing similarity of words using the Wordnet

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))



##############################################################################
################### Text Classification Using NLTK ###########################
##############################################################################

"""
A fairly popular text classification task is to identify a body of text as either
spam or not spam, for things like email filters. In our case, we're going to try to
create a sentiment analysis algorithm.

We'll try to use words as "features" which are a part of either a positive or negative
movie review. The NLTK corpus movie_reviews data set has the reviews, and they are labeled 
already as positive or negative. This means we can train and test with this data.

"""

import nltk
import random
from nltk.corpus import movie_reviews

#movie_reviews.categories() : [u'neg', u'pos']
#movie_reviews.fileids('neg') : list of all the files of the category negative

#In each category (we have pos or neg), take all of the file IDs (each review has its own ID), 
#then store the word_tokenized version (a list of words) for the file ID, followed by the 
#positive or negative label in one big list.

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
                 
#All the words in the file                  
print(len(movie_reviews.words('neg/cv996_12447.txt'))) 

print(documents[1])

#We use random to shuffle our documents. This is because we're going to be training and testing.
#If we left them in order, chances are we'd train on all of the negatives, some positives, 
#and then test only against positives. We don't want that, so we shuffle the data.

random.shuffle(documents)


#we want to collect all words that we find, so we can have a massive list of typical words. 
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())                
                 
print((len(movie_reviews.words()))) #1583820

#From here, we can perform a frequency distribution, to then find out the most common words. 
#As you will see, the most popular "words" are actually things like punctuation, "the," "a" 
#and so on, but quickly we get to legitimate words. We intend to store a few thousand of the 
#most popular words, so this shouldn't be a problem.

all_words = nltk.FreqDist(all_words)

#print most commmon words of the distribution
print(all_words.most_common(25))

#converting words to feature vector in NLTK
word_features = list(all_words.keys())[:3000]

#we're going to build a quick function that will find these top 3,000 words in our
#positive and negative documents, marking their presence as either positive or negative:

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
    
#Next, we can print one feature set like:
words = movie_reviews.words('neg/cv000_29416.txt')

#checked if word dissatisfaction is proesesnt in words
'disatisfaction' in words

#print the feature vector of the document
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# we can do this for all of our documents, saving the feature existence booleans 
#and their respective positive or negative categories by doing:

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#now that we have our features and labels
"""
 Time to choose an algorithm, separate our data into training and testing sets, and press go! 
 The algorithm that we're going to use first is the Naive Bayes classifier. 
 we've shuffled our data set, we'll assign the first 1,900 shuffled reviews, consisting of both positive and negative reviews, as the training set. 
 Then, we can test against the last 100 to see how accurate we are.
 
 Naive Bayes Classifier
 
 Posterior = (Prior * Likelihood)/evidence
 
 P(Class/vector) = ( P(Class) * P(vector/class) )/P(X)
"""

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

#just simply are invoking the Naive Bayes classifier, then we go ahead and use .train() 
#to train it all in one line. Easy enough, now it is trained. Next, we can test it:

classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
classifier.classify(testing_set[0][0])
classifier.classify(testing_set[1][0])

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15) 

"""
train = [
...     (dict(a=1,b=1,c=1), 'y'),
...     (dict(a=1,b=1,c=1), 'x'),
...     (dict(a=1,b=1,c=0), 'y'),
...     (dict(a=0,b=1,c=1), 'x'),
...     (dict(a=0,b=1,c=1), 'y'),
...     (dict(a=0,b=0,c=1), 'y'),
...     (dict(a=0,b=1,c=0), 'x'),
...     (dict(a=0,b=0,c=0), 'x'),
...     (dict(a=0,b=1,c=1), 'y'),
...     ]

test = [
...     (dict(a=1,b=0,c=1)), # unseen
...     (dict(a=1,b=0,c=0)), # unseen
...     (dict(a=0,b=1,c=1)), # seen 3 times, labels=y,y,x
...     (dict(a=0,b=1,c=0)), # seen 1 time, label=x
...     ]

classifier = nltk.classify.NaiveBayesClassifier.train(train)

sorted(classifier.labels())
Out[84]: ['x', 'y']

classifier.classify_many(test)
Out[85]: ['y', 'x', 'y', 'x']
"""

##########################################################################
######################### Saving the classifier ##########################
##########################################################################

import pickle

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

###########################################################################
####################### Different Machine Learing Classifier ##############
###########################################################################

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))


BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))

###########################################################################

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

"""
Now that we know how to use a bunch of algorithmic classifiers, like a child in the candy isle, 
told they can only pick one, we may find it difficult to choose just one classifier. The good 
news is, you don't have to! Combining classifier algorithms is is a common technique, done by 
creating a sort of voting system, where each algorithm gets one vote, and the classification 
that has the votes votes is the chosen one.

"""

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    #We're calling our class the VoteClassifier, and we're inheriting from NLTK's ClassifierI. 
    #Next, we're assigning the list of classifiers that are passed to our class 
    #to self._classifiers.
    def __init__(self, *classifiers):
        self._classifiers = classifiers  
    
    #Easy enough, all we're doing here is iterating through our list of classifier objects. 
    #Then, for each one, we ask it to classify based on the features. The classification is 
    #being treated as a vote. After we are done iterating, we then return the mode(votes), 
    #which is just returning the most popular vote.
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)          
                 

































































