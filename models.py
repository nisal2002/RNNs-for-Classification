# models.py

import numpy as np
import collections
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def predict(self, context):
        value_array=[]
        for char in context:
            index = self.vocab_index.index_of(char)
            value_array.append(index)
        X_tensor = torch.tensor(value_array, dtype=torch.float32)
        output = ConsonantVowelClassifier(X_tensor)
        output_class=np.argmax(output,axis=1)
        return output_class


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)
def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    X=[]
    y=[]

    vowels = ['a', 'e', 'i', 'o', 'u']
    for i in range(0,len(train_cons_exs)-2):
        string = train_cons_exs[i]
        value_array=[]
        for char in string:
            index = vocab_index.index_of(char)
            value_array.append(index)
        X.append(value_array)
        
        if vowels.__contains__(train_cons_exs[i+1][0]):
            y.append([0])
        else:
            y.append([1])
    for i in range(0,len(train_vowel_exs)-2):
        string = train_cons_exs[i]
        value_array=[]
        for char in string:
            index = vocab_index.index_of(char)
            value_array.append(index)
        X.append(value_array)
        
        if vowels.__contains__(train_cons_exs[i+1][0]):
            y.append([0])
        else:
            y.append([1])
    Xn= np.array(X)
    yn = np.array(y)
    time_steps=10
    n_features =20
    X_rnn, y_rnn = create_dataset(Xn, yn,time_steps)
    X_reshaped = X_rnn.reshape(-1,n_features)
    torch.manual_seed(42)
    
    n_output=2

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back to (samples, timesteps, features)
    X_scaled = X_scaled.reshape(len(X_rnn), time_steps, n_features)

    X_train, X_test, y_train, y_test = train_test_split(X_rnn, y_rnn, test_size=0.2, random_state=42,shuffle=True)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNModel, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            out = out[:, -1, :] 
            out = self.fc(out)  
            return out

    hidden_size = 50 
    epochs = 50   
    learning_rate = 0.001
    model = RNNModel(input_size=n_features, hidden_size=hidden_size, output_size=n_output)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(X_train_tensor)
        print(outputs.shape)
        print(y_train_tensor.shape)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

    predictions = test_outputs.numpy()
    print("Predictions for first 5 samples:\n", predictions[:5])    
    class_labels = np.argmax(predictions, axis=1) 
    print(class_labels)
    



    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
