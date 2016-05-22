# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:33:02 2016

@author: Pokemon
An alternative to NLTK's named entity recognition (NER) classifier is provided by the Stanford NER tagger. This tagger is largely seen as the standard in named
entity recognition, but since it uses an advanced statistical learning algorithm it's more computationally expensive than the option provided by NLTK.

A big benefit of the Stanford NER tagger is that is provides us with a few different models for pulling out named entities. We can use any of the following:

3 class model for recognizing locations, persons, and organizations
4 class model for recognizing locations, persons, organizations, and miscellaneous entities
7 class model for recognizing locations, persons, organizations, times, money, percents, and dates


################################################################################################

The parameters passed to the StanfordNERTagger class include:

Classification model path (3 class model used below)
Stanford tagger jar file path
Training data encoding (default of ASCII)

"""

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/usr/share/stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)