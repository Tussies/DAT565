import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd

file_path = 'life_expectancy.csv'
df = pd.read_csv(file_path)

#life_expectancy = load_data('life_expectancy.csv')

LifeTrain
