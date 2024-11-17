# models.py

import numpy as np
import collections
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset,TensorDataset
import re
#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(nn.Module):
    def __init__(self):
        super(ConsonantVowelClassifier, self).__init__()
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


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self,embedding_dim, input_size, hidden_size, output_size,vocab_index):
            super(RNNClassifier,self).__init__()
            self.embedding = nn.Embedding(len(vocab_index),embedding_dim)
            self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True,num_layers=1,bidirectional=True)
            self.fc = nn.Linear(hidden_size*2, output_size)
            self.vocab_index = vocab_index

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        # Pass through the GRU
        output, hidden_layer = self.rnn(embedded)  # Shape: (batch_size, sequence_length, hidden_size * 2)

        # Get the last forward and backward hidden states
        if isinstance(hidden_layer, tuple):
            hidden_layer = hidden_layer[0]

        forward_hidden_layer = hidden_layer[-2, :, :]  # Last forward hidden state
        backward_hidden_layer = hidden_layer[-1, :, :]  # Last backward hidden state

        # Concatenate the last hidden states
        final_hidden_layer = torch.cat((forward_hidden_layer, backward_hidden_layer), dim=1)

        # Pass the concatenated hidden layer through the fully connected layer
        logits = self.fc(final_hidden_layer)  # Output shape: (batch_size, output_size)

        return logits
    def predict(self, context):
        value_array=[]
        for char in context:
            index = self.vocab_index.index_of(char)
            value_array.append(index)
        
        X_tensor = torch.tensor(value_array, dtype=torch.long).unsqueeze(0)

        # X_tensor = X_tensor.unsqueeze(1)
        output = self.forward(X_tensor)
        output_class = np.argmax(output.detach().numpy(), axis=1)
        print(output_class)
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
    # train_cons_exs = train_cons_exs.lower() 
    # train_cons_exs = re.sub(r'[^a-z ]', ' ', train_cons_exs)
    # train_vowel_exs = train_vowel_exs.lower() 
    # train_vowel_exs = re.sub(r'[^a-z ]', ' ', train_vowel_exs)
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
            y.append([1])
        else:
            y.append([0])
    for i in range(0,len(train_vowel_exs)-2):
        string = train_vowel_exs[i]
        value_array=[]
        for char in string:
            index = vocab_index.index_of(char)
            value_array.append(index)
        X.append(value_array)
        
        if vowels.__contains__(train_vowel_exs[i+1][0]):
            y.append([1])
        else:
            y.append([0])
    Xn= np.array(X)
    yn = np.array(y).flatten()
    time_steps=5
    n_features =20
    X_rnn, y_rnn = create_dataset(Xn, yn,time_steps)
    # X_reshaped = X_rnn.reshape(-1,n_features)
    torch.manual_seed(42)
    
    
    X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.1, random_state=42,shuffle=True)
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # X_train_tensor = X_train_tensor.squeeze(0)
    # class RNNModel(nn.Module):
        

    hidden_size = 256 
    embedding_size=32
    epochs = 50   
    learning_rate = 0.009
    n_output=2
    model = RNNClassifier(embedding_dim=embedding_size,input_size=n_features, hidden_size=hidden_size, output_size=n_output,vocab_index=vocab_index)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)#add lr
    total_loss=0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients

        # # Forward pass
        # outputs = model(X_train_tensor)
        # print(outputs.shape)
        # print(y_train_tensor.shape)
        # loss = criterion(outputs, y_train_tensor)

        # # Backward pass and optimization
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        # print(X_train_tensor.shape)
        logits = model.forward(X_train_tensor)
        loss = criterion(logits,y_train_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted_classes = torch.max(logits, 1)  # Get predicted classes
        correct_predictions = (predicted_classes == y_train_tensor).sum().item()  # Count correct predictions
        total_predictions = y_train_tensor.size(0)  # Total number of samples
        accuracy = correct_predictions / total_predictions  # Calculate accuracy

        # Print loss and accuracy every 10 epochs
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')
        
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        _, predicted_classes = torch.max(test_outputs, 1)  # Get indices of the max log-probability

        # Calculate accuracy
        correct_predictions = (predicted_classes == y_test_tensor).sum().item()
        print(correct_predictions)  # Count correct predictions
        total_predictions = y_test_tensor.size(0)  # Total number of samples
        accuracy = correct_predictions / total_predictions  # Calculate accuracy

        print(f'Test Accuracy: {accuracy * 100:.2f}%') 

    return model
    



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


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

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
    def __init__(self,vocab_index, embedding_dim, hidden_dim):
        super(RNNLanguageModel, self).__init__()
        self.embedding =  nn.Embedding(len(vocab_index), embedding_dim)
        self.model_dec = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(vocab_index))
        # self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        self.eval()  
        # context_indices = [self.vocab_index[char] for char in context]
        context_indices=[]
        for char in context:
            index = self.vocab_index.index_of(char)
            context_indices.append(index)
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0) 
        embedded = self.embedding(context_tensor)  
        rnn_out, _ = self.model_dec(embedded)
        last_output = rnn_out[:, -1, :]  # Shape: (1, hidden_dim)
        logits = self.fc(last_output)  
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)  
        target_idx = self.vocab_index.index_of(next_char)
        return log_probs[0, target_idx].item()  # Convert tensor to float

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0

        for i in range(len(next_chars)):

            current_context = context + next_chars[:i]
            next_char = next_chars[i]  
            log_prob = self.get_log_prob_single(next_char, current_context)
            total_log_prob += log_prob
        
        return total_log_prob
    
    def forward(self, x):
        embedded = self.embedding(x)  
        rnn_out, _ = self.model_dec(embedded) 
        output = self.fc(rnn_out[:, -1, :])  
        return output 

 
def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)  
    embedding_dim = 128 
    hidden_dim = 256  
    num_layers = 1
    batch_size = 32
    learning_rate=0.005
    num_epochs=150
    sequence_length=10

    
    train_text = train_text.lower() 
    train_text = re.sub(r'[^a-z ]', ' ', train_text)

    X=[]
    y=[]
    for i in range(len(train_text)-sequence_length):
        sequence = train_text[i:i+sequence_length]
        target = train_text[i+sequence_length]
        index=[]
        for char in sequence:
            index_of_char = vocab_index.index_of(char)
            index.append(index_of_char)
        target_index = vocab_index.index_of(target)
        X.append(index)
        y.append(target_index)

    
    
    X_tensor =torch.tensor(X, dtype= torch.long) # was dtype = torch.long
    y_tensor = torch.tensor(y,dtype= torch.long)
    dataset = TensorDataset(X_tensor,y_tensor)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    model = RNNLanguageModel(vocab_index, embedding_dim, hidden_dim)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs,targets in data_loader:
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')
    
    return model



