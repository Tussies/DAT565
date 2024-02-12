import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd

def load_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                data.append(content)
    return data

easy_ham = load_data('easy_ham')
hard_ham = load_data('hard_ham')
spam = load_data('spam')

easyHamTrain, easyHamTest = train_test_split(easy_ham, test_size=0.25, random_state=42)
hardHamTrain, hardHamTest = train_test_split(hard_ham, test_size=0.25, random_state=42)
spamTrain, spamTest = train_test_split(spam, test_size=0.25, random_state=42)

X_train = easyHamTrain + hardHamTrain + spamTrain
y_train = [0] * len(easyHamTrain) + [1] * len(hardHamTrain) + [2] * len(spamTrain)

X_test = easyHamTest + hardHamTest + spamTest
y_test = [0] * len(easyHamTest) + [1] * len(hardHamTest) + [2] * len(spamTest)

X_train_hard = hardHamTrain + spamTrain
X_test_hard = hardHamTest + spamTest

y_train_hard = [0] * len(hardHamTrain) + [1] * len(spamTrain)
y_test_hard = [0] * len(hardHamTest) + [1] * len(spamTest)

vectorizer = CountVectorizer()
X_train_vectorized_hard = vectorizer.fit_transform(X_train_hard)
X_test_vectorized_hard = vectorizer.transform(X_test_hard)

classifierMNB = MultinomialNB()
classifierMNB.fit(X_train_vectorized_hard, y_train_hard)

classifierBNB = BernoulliNB()
classifierBNB.fit(X_train_vectorized_hard, y_train_hard)

predictionsMNB = classifierMNB.predict(X_test_vectorized_hard)
predictionsBNB = classifierBNB.predict(X_test_vectorized_hard)
print(predictionsMNB)
print(predictionsBNB)

scoreMNB = classifierMNB.score(X_test_vectorized_hard, y_test_hard)
print("Accuracy for Multinomial Naive Bayes:", scoreMNB)

precisionMNB = precision_score(y_test_hard, predictionsMNB, average='binary', pos_label=1)
print("Precision for Multinomial Naive Bayes:", precisionMNB)

recallMNB = recall_score(y_test_hard, predictionsMNB, average='binary', pos_label=1)
print("Recall for Multinomial Naive Bayes:", recallMNB)


scoreBNB = classifierBNB.score(X_test_vectorized_hard, y_test_hard)
print("Accuracy for Bernoulli Naive Bayes:", scoreBNB)

precisionBNB = precision_score(y_test_hard, predictionsBNB, average='binary', pos_label=1)
print("Precision for Bernoulli Naive Bayes:", precisionBNB)

recallBNB = recall_score(y_test_hard, predictionsBNB, average='binary', pos_label=1)
print("Recall for Bernoulli Naive Bayes:", recallBNB)

conf_matrix_MNB = confusion_matrix(y_test_hard, predictionsMNB)
conf_matrix_df_MNB = pd.DataFrame(conf_matrix_MNB, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_MNB.loc['Total'] = conf_matrix_df_MNB.sum()
conf_matrix_df_MNB['Total'] = conf_matrix_df_MNB.sum(axis=1)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(conf_matrix_df_MNB)

conf_matrix_BNB = confusion_matrix(y_test_hard, predictionsBNB)
conf_matrix_df_BNB = pd.DataFrame(conf_matrix_BNB, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_BNB.loc['Total'] = conf_matrix_df_BNB.sum()
conf_matrix_df_BNB['Total'] = conf_matrix_df_BNB.sum(axis=1)
print("\nConfusion Matrix for Bernoulli Naive Bayes:")
print(conf_matrix_df_BNB)
