import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Sizes of Training and Testing Sets:")
print("Easy Ham: Train =", len(easyHamTrain), "Test =", len(easyHamTest))
print("Hard Ham: Train =", len(hardHamTrain), "Test =", len(hardHamTest))
print("Spam: Train =", len(spamTrain), "Test =", len(spamTest))

total_emails_before_split = len(easy_ham) + len(hard_ham) + len(spam)
total_emails_after_split = len(easyHamTrain) + len(easyHamTest) + len(hardHamTrain) + len(hardHamTest) + len(spamTrain) + len(spamTest)

print("\nTotal number of emails before split:", total_emails_before_split)
print("Total number of emails after split:", total_emails_after_split)

print("\nShape of Vectorized Training Data:", X_train_vectorized.shape)
print("Shape of Vectorized Testing Data:", X_test_vectorized.shape)
