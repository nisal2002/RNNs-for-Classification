#####################
# MODELS FOR PART 2 #
#####################
import numpy as np
import torch
import torch.nn as nn

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
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """


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