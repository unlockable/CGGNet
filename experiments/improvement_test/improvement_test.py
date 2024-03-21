import os
from collections import Counter
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

class SolidityDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(sequence) for sequence in sequences]
        self.labels = labels
        print("len(self.sequences):", len(self.sequences))
        print("len(self.labels):", len(self.labels))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        #print(idx)
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]
        out = self.fc(hidden)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        hidden = gru_out[:, -1, :]
        out = self.fc(hidden)
        return out
    
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        hidden = rnn_out[:, -1, :]
        out = self.fc(hidden)
        return out    

def read_solidity_files(directory):
    solidity_files = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()
            solidity_files.append(content)
    return solidity_files

# Function to evaluate accuracy
def evaluate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    print(correct_predictions)
    accuracy = 100 * correct_predictions / total_predictions
    return accuracy

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")



parser = argparse.ArgumentParser(description="Select classifier")
parser.add_argument('--model', type=str, help="Select model")
parser.add_argument('--expr', type=str, help="Select experiment type")

args = parser.parse_args()

model_type = args.model
expr_type = args.expr

solidity_normal_train_data = read_solidity_files("./normal_train")
train_labels = [0] * 500


solidity_reentrancy_train_data = read_solidity_files("./reentrancy_train")

train_labels = train_labels + [1] * 500
solidity_train_data = solidity_normal_train_data + solidity_reentrancy_train_data



solidity_normal_test_data = read_solidity_files("./normal_evaluation")
test_normal_labels = [0] * 1000

solidity_reentrancy_test_data = read_solidity_files("./reentrancy_evaluation")
test_labels = test_normal_labels + [1] * 1000

solidity_test_data = solidity_normal_test_data + solidity_reentrancy_test_data



solidity_normal_augmented_data = read_solidity_files("./normal_augmented")
augmented_normal_labels = [0] * 10000

solidity_reentrancy_augmented_data = read_solidity_files("./reentrancy_augmented")
augmented_labels = augmented_normal_labels + [1] * 10000

solidity_aug_data = solidity_normal_augmented_data + solidity_reentrancy_augmented_data

solidity_aug_train_data = solidity_aug_data + solidity_train_data
augmented_train_labels = augmented_labels + train_labels

print("solidity_aug_train_data:", len(solidity_aug_train_data))
print("augmented_train_labels:", len(augmented_train_labels))

total_solidity_data = solidity_aug_data + solidity_test_data + solidity_train_data
def tokenize(solidity_code):
    # Basic tokenization by space - can be improved
    return solidity_code.split()

def create_vocab(solidity_files):
    tokens = set()
    for file in solidity_files:
        tokens.update(tokenize(file))
    # Add a special token for unknown words
    tokens.add("<UNK>")
    return {token: i for i, token in enumerate(tokens)}

def encode(tokenized_code, vocab, max_length):
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokenized_code]
    padded = encoded + [vocab.get("<PAD>")] * (max_length - len(encoded))  # Assuming <PAD> is in your vocab
    return padded

# Tokenize and encode Solidity data
vocab = create_vocab(total_solidity_data)
vocab_size = len(vocab) + 1

print(vocab_size)
vocab["<PAD>"] = len(vocab)  # Assign a unique index to the padding token

if expr_type == "aug":
    encoded_data = [encode(tokenize(file), vocab, 327) for file in solidity_aug_train_data]
else :
    encoded_data = [encode(tokenize(file), vocab, 327) for file in solidity_train_data]
test_encoded_data = [encode(tokenize(file), vocab, 327) for file in solidity_test_data]

print("test_encoded_data", len(test_encoded_data))
max_length = max(len(sequence) for sequence in encoded_data)  # or set a predefined length

#train_data, test_data, train_labels, test_labels = train_test_split(encoded_data, train_labels, test_size=0.2)
if expr_type == "aug":
    train_dataset = SolidityDataset(encoded_data, augmented_train_labels)
else :
    train_dataset = SolidityDataset(encoded_data, train_labels)
test_dataset = SolidityDataset(test_encoded_data, test_labels)
# Assuming `data` is your tokenized and encoded Solidity code and `labels` are your labels
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

embedding_dim = 128  # Size of the embedding vector
hidden_dim = 256     # Number of features in the hidden state of the LSTM
output_dim = 2       # For binary classification (vulnerability present or not)


#model = FCClassifier()
if model_type == "lstm":
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
elif model_type == "rnn":
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
if model_type == "gru":
    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

print("model tyep :", model_type)

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

train_losses = []
test_losses = []
import matplotlib.pyplot as plt

total_test_loss_all_runs = []

for test_time in range(50):
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Evaluate on test data
        model.eval()  # Set model to evaluation mode
        total_test_loss = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        average_test_loss = total_test_loss / len(test_loader)
        test_losses.append(average_test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}')
    
    total_test_loss_all_runs.append(test_losses)

# Now, total_test_loss_all_runs is a list of lists containing the test losses for each run

print("total_test_loss_all_runs:", total_test_loss_all_runs)
# Plotting the average test loss

average_loss_per_epoch = np.mean(total_test_loss_all_runs, axis=0)

# Plotting the average loss
plt.figure(figsize=(10, 6))
plt.plot(average_loss_per_epoch, label='Average Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Test Loss Per Epoch')
plt.legend()
plt.show()
plt.savefig('loss_plot_50_before.png')  # Saves the plot as a PNG file


# Call the function to evaluate the model


