# models.py

import numpy as np
import collections
import torch as torch
import torch.nn as nn
import torch.optim as optim


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


class RNNClassifier(ConsonantVowelClassifier, nn.Module):

    def __init__(self, embedding_layer, rnn, output, vocabulary, context_length):

        super(RNNClassifier, self).__init__()
        self.embedding_layer = embedding_layer  # embedding layer to convert input characters to dense vectors
        self.rnn = rnn
        self.output = output
        self.vocabulary = vocabulary
        self.context_length = context_length  # context length for input sequence

    def convert_to_tensor(self, string):
        # convert the input string to a tensor using the vocabulary
        indices = [self.vocabulary.index_of(character) for character in string]
        tensor = torch.tensor(indices, dtype=torch.long)
        tensor = tensor.unsqueeze(0)  # batch dimension (shape: [1, sequence_length])
        return tensor

    def forward_pass(self, x):

        embedded = self.embedding_layer(x)  # pass input tensor through the embedding layer
        output, hidden_layer = self.rnn(embedded)

        if isinstance(hidden_layer, tuple):
            hidden_layer = hidden_layer[0]

        forward_hidden_layer = hidden_layer[-2, :, :]  # last forward hidden state (for bidirectional RNN)
        backward_hidden_layer = hidden_layer[-1, :, :]  # last backward hidden state (for bidirectional RNN)


        final_hidden_layer = torch.cat((forward_hidden_layer, backward_hidden_layer), dim=1)


        logits = self.output(final_hidden_layer)
        return logits

    def predict(self, context):
        # convert input string to tensor if string
        if isinstance(context, str):
            context = self.convert_to_tensor(context)
        else:
            pass


        logits = self.forward_pass(context)

        pred = torch.argmax(logits, dim=1).item()
        return pred



def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    vocabulary_size = len(vocab_index)  # size of vocabulary
    embedding_dimension = 32
    hidden_dimension = 64
    context_length = 20
    epochs = 20
    learning_rate = 0.001


    embedding_layer = nn.Embedding(vocabulary_size, embedding_dimension)  # embedding layer to convert inputs to vectors
    rnn = nn.GRU(embedding_dimension, hidden_dimension, num_layers=2, batch_first=True, bidirectional=True)
    output = nn.Linear(hidden_dimension * 2, 2)  # classify into consonant or vowel

    # RNN model
    model = RNNClassifier(embedding_layer, rnn, output, vocab_index, context_length=context_length)
    print(model)
    loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        model.train()
        total_loss = 0


        for cons_ex, vowel_ex in zip(train_cons_exs, train_vowel_exs):
            consonent_tensor = model.convert_to_tensor(cons_ex)
            vowel_tensor = model.convert_to_tensor(vowel_ex)


            optimizer.zero_grad()  # zero the gradients
            logits = model.forward_pass(consonent_tensor)
            loss = loss_function(logits, torch.tensor([0], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Training on vowel examples
            optimizer.zero_grad()
            logits = model.forward_pass(vowel_tensor)
            loss = loss_function(logits, torch.tensor([1], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_cons_exs)
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")


        model.eval()
        with torch.no_grad():  # offing gradient calculation for validation
            validation_loss = 0
            correct_predictions = 0
            total_predictions = 0


            for cons_ex, vowel_ex in zip(dev_cons_exs, dev_vowel_exs):
                consonent_tensor = model.convert_to_tensor(cons_ex)
                vowel_tensor = model.convert_to_tensor(vowel_ex)


                logits = model.forward_pass(consonent_tensor)
                validation_loss += loss_function(logits, torch.tensor([0], dtype=torch.long))
                prediction = torch.argmax(logits, dim=1)
                correct_predictions += (prediction == 0).sum().item()
                total_predictions += 1


                logits = model.forward_pass(vowel_tensor)
                validation_loss += loss_function(logits, torch.tensor([1], dtype=torch.long))
                prediction = torch.argmax(logits, dim=1)
                correct_predictions += (prediction == 1).sum().item()
                total_predictions += 1


            average_validation_loss = validation_loss / (len(dev_cons_exs) + len(dev_vowel_exs))
            accuracy = correct_predictions / total_predictions
            print(f"Validation Loss: {average_validation_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")  # Print validation results

    return model



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