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


#Problem 1B & 2
easyHamTrain, easyHamTest = train_test_split(easy_ham, test_size=0.25, random_state=42)
hardHamTrain, hardHamTest = train_test_split(hard_ham, test_size=0.25, random_state=42)
spamTrain, spamTest = train_test_split(spam, test_size=0.25, random_state=42)

X_train_easy = easyHamTrain + spamTrain
X_test_easy = easyHamTest + spamTest

y_train_easy = [0] * len(easyHamTrain) + [1] * len(spamTrain)
y_test_easy = [0] * len(easyHamTest) + [1] * len(spamTest)

vectorizer = CountVectorizer()
X_train_vectorized_easy = vectorizer.fit_transform(X_train_easy)
X_test_vectorized_easy = vectorizer.transform(X_test_easy)

X_train_hard = hardHamTrain + spamTrain
X_test_hard = hardHamTest + spamTest

y_train_hard = [1] * len(hardHamTrain) + [2] * len(spamTrain)
y_test_hard = [1] * len(hardHamTest) + [2] * len(spamTest)

X_train_vectorized_hard = vectorizer.transform(X_train_hard)
X_test_vectorized_hard = vectorizer.transform(X_test_hard)

#Problem 3 & 4
classifierMNB = MultinomialNB()
classifierMNB.fit(X_train_vectorized_easy, y_train_easy)

classifierBNB = BernoulliNB()
classifierBNB.fit(X_train_vectorized_easy, y_train_easy)

predictionsMNB = classifierMNB.predict(X_test_vectorized_easy)
predictionsBNB = classifierBNB.predict(X_test_vectorized_easy)
print(predictionsMNB)
print(predictionsBNB)

classifierMNB_hard = MultinomialNB()
classifierMNB_hard.fit(X_train_vectorized_hard, y_train_hard)

classifierBNB_hard = BernoulliNB()
classifierBNB_hard.fit(X_train_vectorized_hard, y_train_hard)

predictionsMNB_hard = classifierMNB_hard.predict(X_test_vectorized_hard)
predictionsBNB_hard = classifierBNB_hard.predict(X_test_vectorized_hard)
print(predictionsMNB_hard)
print(predictionsBNB_hard)

scoreMNB = classifierMNB.score(X_test_vectorized_easy, y_test_easy)
print("Accuracy for Multinomial Naive Bayes:", scoreMNB)

scoreBNB = classifierBNB.score(X_test_vectorized_easy, y_test_easy)
print("Accuracy for Bernoulli Naive Bayes:", scoreBNB)

precisionMNB = precision_score(y_test_easy, predictionsMNB, average='binary', pos_label=1)
print("Precision for Multinomial Naive Bayes:", precisionMNB)

precisionBNB = precision_score(y_test_easy, predictionsBNB, average='binary', pos_label=1)
print("Precision for Bernoulli Naive Bayes:", precisionBNB)

recallMNB = recall_score(y_test_easy, predictionsMNB, average='binary', pos_label=1)
print("Recall for Multinomial Naive Bayes:", recallMNB)

recallBNB = recall_score(y_test_easy, predictionsBNB, average='binary', pos_label=1)
print("Recall for Bernoulli Naive Bayes:", recallBNB)

conf_matrix_MNB = confusion_matrix(y_test_easy, predictionsMNB)
conf_matrix_df_MNB = pd.DataFrame(conf_matrix_MNB, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_MNB.loc['Total'] = conf_matrix_df_MNB.sum()
conf_matrix_df_MNB['Total'] = conf_matrix_df_MNB.sum(axis=1)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(conf_matrix_df_MNB)

conf_matrix_BNB = confusion_matrix(y_test_easy, predictionsBNB)
conf_matrix_df_BNB = pd.DataFrame(conf_matrix_BNB, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_BNB.loc['Total'] = conf_matrix_df_BNB.sum()
conf_matrix_df_BNB['Total'] = conf_matrix_df_BNB.sum(axis=1)
print("\nConfusion Matrix for Bernoulli Naive Bayes:")
print(conf_matrix_df_BNB)

scoreMNB_hard = classifierMNB_hard.score(X_test_vectorized_hard, y_test_hard)
print("\nAccuracy for Multinomial Naive Bayes with Hard Ham:", scoreMNB_hard)

precisionMNB_hard = precision_score(y_test_hard, predictionsMNB_hard, average='binary', pos_label=2)
print("Precision for Multinomial Naive Bayes with Hard Ham:", precisionMNB_hard)

recallMNB_hard = recall_score(y_test_hard, predictionsMNB_hard, average='binary', pos_label=2)
print("Recall for Multinomial Naive Bayes with Hard Ham:", recallMNB_hard)

conf_matrix_MNB_hard = confusion_matrix(y_test_hard, predictionsMNB_hard)
conf_matrix_df_MNB_hard = pd.DataFrame(conf_matrix_MNB_hard, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_MNB_hard.loc['Total'] = conf_matrix_df_MNB_hard.sum()
conf_matrix_df_MNB_hard['Total'] = conf_matrix_df_MNB_hard.sum(axis=1)
print("\nConfusion Matrix for Multinomial Naive Bayes with Hard Ham:")
print(conf_matrix_df_MNB_hard)

scoreBNB_hard = classifierBNB_hard.score(X_test_vectorized_hard, y_test_hard)
print("\nAccuracy for Bernoulli Naive Bayes with Hard Ham:", scoreBNB_hard)

precisionBNB_hard = precision_score(y_test_hard, predictionsBNB_hard, average='binary', pos_label=2)
print("Precision for Bernoulli Naive Bayes with Hard Ham:", precisionBNB_hard)

recallBNB_hard = recall_score(y_test_hard, predictionsBNB_hard, average='binary', pos_label=2)
print("Recall for Bernoulli Naive Bayes with Hard Ham:", recallBNB_hard)

conf_matrix_BNB_hard = confusion_matrix(y_test_hard, predictionsBNB_hard)
conf_matrix_df_BNB_hard = pd.DataFrame(conf_matrix_BNB_hard, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
conf_matrix_df_BNB_hard.loc['Total'] = conf_matrix_df_BNB_hard.sum()
conf_matrix_df_BNB_hard['Total'] = conf_matrix_df_BNB_hard.sum(axis=1)
print("\nConfusion Matrix for Bernoulli Naive Bayes with Hard Ham:")
print(conf_matrix_df_BNB_hard)