#Imports
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('punkt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections as cl
import glob

import sklearn

easy_ham = 'easy_ham'
hard_ham = 'hard_ham'
spam = 'spam'

df = pd.read_html('easy_ham')

#Split data into train and test sets
easyHamTrain, easyHamTest = train_test_split(easy_ham, test_size=0.25)
hardHamTrain, hardHamTest = train_test_split(hard_ham, test_size=0.25)
spamTrain, spamTest = train_test_split(spam, test_size=0.25)
