import pandas as pd
import nltk
import string 
import argparse
import os
import pickle
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')


class SVMClassifier:
    '''
    This classifier is trained using a Support Vector Machine (SVM)
    The data are preprocessed using lemmatization, stopwords and punctuation
    are removed. The features are extracted using TF-IDF.
    '''

    def __init__(self, training_examples: List):
        X = training_examples["Text"].tolist()
        y = training_examples["Emotion"].tolist()

        X = self._lemmatize(X)
        X = self._remove_stopwords_and_punctuation(X)

        self.vectorizer = TfidfVectorizer()
        X_transformed = self.vectorizer.fit_transform(X)

        self.classifier = SVC(kernel='linear', probability=True)
        self.classifier.fit(X_transformed, y)

    def _lemmatize(self, sentences):
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in sentence.split()]) for sentence in sentences]

    def _remove_stopwords_and_punctuation(self, sentences):
        stopwords_set = set(stopwords.words('english'))
        punctuation_set = set(string.punctuation)
        return [' '.join([word.lower() for word in word_tokenize(sentence) if 
                          (word.lower() not in stopwords_set and word.lower() not in punctuation_set)]) 
                          for sentence in sentences]

    def classify(self, text):
        text = self._lemmatize([text])[0]
        text = self._remove_stopwords_and_punctuation([text])[0]
        vectorized_text = self.vectorizer.transform([text])

        return self.classifier.predict(vectorized_text)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a baseline SVM classifier')
    parser.add_argument('train_dataset_path', type=str, help='Path to the training dataset')
    parser.add_argument('test_dataset_path', type=str, help='Path to the test dataset')
    args = parser.parse_args()

    train_dataset_path = args.train_dataset_path
    if not os.path.exists(train_dataset_path):
        print(f"Training dataset file does not exist: {train_dataset_path}")
        exit()

    test_dataset_path = args.test_dataset_path
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset file does not exist: {test_dataset_path}")
        exit()

    # here I can decide which test dataset to use (test or validation)
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    # instantiate and train the SVMClassifier
    svm_classifier = SVMClassifier(train_data)

    # classify the test data and compute the accuracy score, F1 score and MCC
    y_true = test_data["Emotion"].tolist()
    y_pred = [svm_classifier.classify(text) for text in test_data["Text"].tolist()]
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    matthews = matthews_corrcoef(y_true, y_pred)

    print(f"Accuracy score: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Matthews correlation coefficient: {matthews:.4f}")

    # save the model
    with open('svm_classifier.pkl', 'wb') as file:
        pickle.dump(svm_classifier, file)