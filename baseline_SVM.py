import pandas as pd
import nltk
import string 
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
    The data are preprocessed using lemmatization, removing stopwords and punctuation
    '''

    def __init__(self, training_examples: List):
        # convert the data into lists
        X = training_examples["Text"].tolist()
        y = training_examples["Emotion"].tolist()

        # perform lemmatization on the input data
        X = self._lemmatize(X)

        # remove stopwords and punctuation
        X = self._remove_stopwords_and_punctuation(X)

        # convert the text into numerical values using TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        X_transformed = self.vectorizer.fit_transform(X)

        # create a classifier and train it on the data
        self.classifier = SVC(kernel='linear', probability=True)
        self.classifier.fit(X_transformed, y)

    def _lemmatize(self, sentences):
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in sentence.split()]) for sentence in sentences]

    def _remove_stopwords_and_punctuation(self, sentences):
        stopwords_set = set(stopwords.words('english'))
        punctuation_set = set(string.punctuation)
        return [' '.join([word.lower() for word in word_tokenize(sentence) if (word.lower() not in stopwords_set and word.lower() not in punctuation_set)]) for sentence in sentences]

    def classify(self, text):
        # perform lemmatization on the input text
        text = self._lemmatize([text])[0]

        # remove stopwords and punctuation
        text = self._remove_stopwords_and_punctuation([text])[0]

        # Vectorizing the text using the same vectorizer
        vectorized_text = self.vectorizer.transform([text])
        # Classifying the vectorized text:
        return self.classifier.predict(vectorized_text)
    

if __name__ == '__main__':

    train_data = pd.read_csv('Datasets/data_train.csv')
    val_data = pd.read_csv('Datasets/data_val.csv')
    test_data = pd.read_csv('Datasets/data_test.csv')

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