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

#pre-processing code here

#Create function to convert data
def get_data(dir_path):
  mail_list = []
  for mail_path in glob.glob(dir_path):
    with open(mail_path, "r", encoding='iso-8859-1') as mail:
      try:
        mail_list.append(mail.read())

      except:
        print("Couldn't read file", mail_path, "from", dir_path)
  return(mail_list)

#Load data
easy_ham = get_data("./easy_ham/*")
hard_ham = get_data("./hard_ham/*")
spam = get_data("./spam/*")

#Combine data
combinedData = easy_ham + hard_ham + spam

vectorizer = CountVectorizer()
vectorizer.fit(combinedData)

#Split data into train and test sets
easyHamTrain, easyHamTest = train_test_split(easy_ham, test_size=0.3)
hardHamTrain, hardHamTest = train_test_split(hard_ham, test_size=0.3)
spamTrain, spamTest = train_test_split(spam, test_size=0.3)