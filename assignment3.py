import os
from sklearn.model_selection import train_test_split

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