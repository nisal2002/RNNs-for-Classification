#####################
# MODELS FOR PART 2 #
#####################
"""
import numpy as np
import torch
import torch.nn as nn

class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_dim), hidden
        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, seq_len, vocab_size)
        return output, hidden

    def init_hidden(self, batch_size, hidden_dim):
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))

    def get_log_prob_single(self, next_char, context):
        # Forward pass with the context and predict log prob for next_char
        self.eval()
        with torch.no_grad():
            context_idx = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
            hidden = self.init_hidden(1, self.rnn.hidden_size)
            output, _ = self(context_idx, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            log_prob = torch.log(probs[self.vocab_index.index_of(next_char)])
            return log_prob.item()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        for i, char in enumerate(next_chars):
            log_prob += self.get_log_prob_single(char, context + next_chars[:i])
        return log_prob

    # def train_lm(args, train_text, dev_text, vocab_index):
    
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    


#    raise Exception("Implement me")
def train_lm(args, train_text, dev_text, vocab_index):
    # Hyperparameters
    embedding_dim = 50
    hidden_dim = 100
    batch_size = 64
    seq_length = 30
    num_epochs = 10
    lr = 0.001

    # Prepare data
    train_indices = [vocab_index.index_of(c) for c in train_text]
    dev_indices = [vocab_index.index_of(c) for c in dev_text]

    def create_batches(data, seq_length, batch_size):
        sequences = []
        for i in range(0, len(data) - seq_length):
            sequences.append(data[i:i + seq_length + 1])
        inputs = [seq[:-1] for seq in sequences]
        targets = [seq[1:] for seq in sequences]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    train_inputs, train_targets = create_batches(train_indices, seq_length, batch_size)
    dev_inputs, dev_targets = create_batches(dev_indices, seq_length, batch_size)

    # Initialize the model
    model = RNNLanguageModel(len(vocab_index), embedding_dim, hidden_dim, vocab_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(train_inputs) - seq_length, batch_size):
            inputs = torch.tensor([train_indices[i:i + seq_length]], dtype=torch.long)
            targets = torch.tensor([train_indices[i + 1:i + seq_length + 1]], dtype=torch.long)

            # Adjust hidden state size based on batch size
            current_batch_size = inputs.size(0)
            hidden = model.init_hidden(current_batch_size, hidden_dim)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(outputs.view(-1, len(vocab_index)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print epoch progress
        avg_loss = total_loss / len(train_inputs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # Evaluate on dev set
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(len(dev_inputs), hidden_dim)
        outputs, _ = model(dev_inputs, hidden)
        dev_loss = criterion(outputs.view(-1, len(vocab_index)), dev_targets.view(-1))
        print(f"Dev Loss: {dev_loss.item():.4f}")

    return model

"""
import numpy as np
import collections
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import re
import matplotlib.pyplot as plt


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
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_index, embedding_dim, hidden_dim,dropout_prob=0.5):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab_index), embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.model_dec = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(vocab_index))
        self.vocab_index = vocab_index


    def get_log_prob_single(self, next_char, context):
        self.eval()
        # context_indices = [self.vocab_index[char] for char in context]
        context_indices = []
        for char in context:
            index = self.vocab_index.index_of(char)
            context_indices.append(index)
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
        embedded = self.embedding(context_tensor)
        embedded = self.dropout(embedded)
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
        embedded = self.dropout(embedded)  # Dropout to embedding
        rnn_out, _ = self.model_dec(embedded)
        rnn_out = self.dropout(rnn_out)  # Dropout to RNN output
        output = self.fc(rnn_out[:, -1, :])
        return output


def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 1
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    sequence_length = 20

    train_text = train_text.lower()
    train_text = re.sub(r'[^a-z ]', ' ', train_text)

    X = []
    y = []
    for i in range(len(train_text) - sequence_length):
        sequence = train_text[i:i + sequence_length]
        target = train_text[i + sequence_length]
        index = []
        for char in sequence:
            index_of_char = vocab_index.index_of(char)
            index.append(index_of_char)
        target_index = vocab_index.index_of(target)
        X.append(index)
        y.append(target_index)

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RNNLanguageModel(vocab_index, embedding_dim, hidden_dim)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')  # Initialize best loss to a very high value.
    patience = 5  # Number of epochs to wait for improvement.
    patience_counter = 0

    # List to store values for plotting
    epoch_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)  # Get index of max logit
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(data_loader)
        epoch_losses.append(avg_loss)
        accuracy = correct / total * 100  # Convert to percentage

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        for i in range(len(dev_text) - sequence_length):
            context = dev_text[i:i + sequence_length]
            next_char = dev_text[i + sequence_length]
            context_indices = [vocab_index.index_of(c) for c in context]
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
            target_index = vocab_index.index_of(next_char)
            output = model(context_tensor)
            val_loss = criterion(output, torch.tensor([target_index]))
            total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dev_text)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch + 1} due to no improvement.")
            break

    # Evaluation: Calculate Perplexity and Likelihood
    model.eval()
    total_log_likelihood = 0.0
    total_chars = 0

    for i in range(len(dev_text) - sequence_length):
        context = dev_text[i:i + sequence_length]
        next_char = dev_text[i + sequence_length]
        log_prob = model.get_log_prob_single(next_char, context)
        total_log_likelihood += log_prob
        total_chars += 1

    # Compute Perplexity
    avg_log_likelihood = total_log_likelihood / total_chars
    perplexity = np.exp(-avg_log_likelihood)
    print(f'Log-Likelihood: {total_log_likelihood:.4f}')
    print(f'Perplexity: {perplexity:.4f}')

    # Plot the training and validation loss over epochs
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # Add a legend to differentiate between the two lines
    plt.show()

    return model
