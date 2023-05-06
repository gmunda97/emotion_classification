import pandas as pd
import numpy as np
import nltk
import string
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class TextClassificationDataset(Dataset):
    def __init__(self, df, vectorizer, label_encoder):
        self.df = df
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token not in self.stop_words and token not in self.punctuation]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        preprocess_text = " ".join(tokens)
        return preprocess_text
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]["Text"]
        label = self.df.iloc[idx]["Emotion"]

        text = self.preprocess_text(text)
        bow = self.vectorizer.transform([text]).toarray()[0]
        label = self.label_encoder.transform([label])[0]
        
        return torch.LongTensor([label]), torch.FloatTensor(bow)
    

class FullyConnectedNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
    

if __name__ == '__main__':

    train_df = pd.read_csv("Datasets/data_train.csv")
    eval_df = pd.read_csv("Datasets/data_val.csv")
    test_df = pd.read_csv("Datasets/data_test.csv")

    all_labels = np.concatenate([train_df["Emotion"].unique(), eval_df["Emotion"].unique()])
    unique_labels = np.unique(all_labels)
    sorted_labels = np.sort(unique_labels)

    vectorizer = CountVectorizer()
    label_encoder = LabelEncoder()

    vectorizer.fit(train_df["Text"])
    label_encoder.fit(train_df["Emotion"])

    train_dataset = TextClassificationDataset(train_df, vectorizer, label_encoder)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    eval_labels = eval_df["Emotion"]
    eval_label_encoder = LabelEncoder()
    eval_label_encoder.fit(eval_labels)

    eval_dataset = TextClassificationDataset(eval_df, vectorizer, eval_label_encoder)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=True)

    test_labels = test_df["Emotion"]
    test_label_encoder = LabelEncoder()
    test_label_encoder.fit(test_labels)

    test_dataset = TextClassificationDataset(test_df, vectorizer, test_label_encoder)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    net = FullyConnectedNeuralNetwork(len(vectorizer.vocabulary_), 20, 6, 0.2)

    # define loss function and optimizer
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_accuracy = 0
    best_model = None

    # Initialize lists to store loss and accuracy values
    train_losses, val_losses = [], []
    val_accs, test_accs = [], []

    # Early stopping parameters
    patience = 3
    counter = 0 
    best_loss = float('inf')

    # Train model
    for epoch in range(8):
        total_loss, total_acc, count = 0, 0, 0
        net.train()
        for labels, features in train_dataloader:
            optimizer.zero_grad()
            out = net(features)
            labels = labels.squeeze(1)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: training loss = {total_loss/len(train_dataset)}")

        # Evaluate model on validation set
        net.eval()
        total_correct = 0
        val_loss = 0
        with torch.no_grad():
            for labels, features in eval_dataloader:
                out = net(features)
                labels = labels.squeeze(1)
                _, predicted = torch.max(out, dim=1)
                total_correct += (predicted == labels).sum().item()
                val_loss += loss_fn(out, labels).item()
            val_accuracy = total_correct / len(eval_dataset)
            val_loss /= len(eval_dataset)
            
        print(f"Accuracy: {val_accuracy}, Validation loss: {val_loss}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model = net.state_dict()
            torch.save(best_model, "best_model_prova.pt")
            counter = 0 # Reset counter since we have seen improvement
        else:
            counter += 1 # No improvement, increase counter
            if counter >= patience:
                print(f"No improvement after {patience} epochs. Stopping training.")
                break

        # Evaluate model on test set
        net.load_state_dict(best_model)
        net.eval()
        total_correct = 0
        with torch.no_grad():
            for labels, features in test_dataloader:
                out = net(features)
                labels = labels.squeeze(1)
                _, predicted = torch.max(out, dim=1)
                total_correct += (predicted == labels).sum().item()
            test_accuracy = total_correct / len(test_dataset)

        print(f"Test accuracy: {test_accuracy}")

        # Save loss and accuracy values
        train_losses.append(total_loss / len(train_dataset))
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        test_accs.append(test_accuracy)