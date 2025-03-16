import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------#
# Data Preprocessing and Caching #
# -------------------------------#
def load_or_cache_data(data_dir, classes, target_shape=(128,150), 
                       cache_file='processed_cache/cached_data_lstm.pt'):
    """
    If data was preprocessed into a file, load it. Otherwise, process the data.
    """
    if os.path.exists(cache_file):
        print(f"Loading preprocessed data from {cache_file}...")
        data_tensor, labels_tensor = torch.load(cache_file, weights_only=False)
    else:
        print("Preprocessing data...")
        from processing import load_audio_lstm
        data_tensor, labels_tensor = load_audio_lstm(data_dir, classes)
        print(f"Saving preprocessed data to {cache_file}...")
        torch.save((data_tensor, labels_tensor), cache_file)
    
    return data_tensor, labels_tensor

# -------------------------------#
#       Dataset Definition       #
# -------------------------------#

class GenreDataset(Dataset):
    def __init__(self, root_dir, classes, target_shape=(128,150), transform=None):
        """
        Args:
            root_dir (str): Directory containing genre subfolders.
            classes (list): List of genre names.
            target_shape (tuple): Desired shape (frequency, time) for spectrograms.
            transform (callable, optional): Optional transform to be applied.
        """
        self.transform = transform
        data, labels = load_or_cache_data(root_dir, classes, target_shape)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        # Original sample shape: (1, frequency, time)
        sample = self.data[idx]
        label = self.labels[idx]
        # For LSTM, we want a shape of (time, features) where:
        # time = target_shape[1] and features = target_shape[0].
        # Squeeze out the channel dimension and transpose.
        #sample = sample.squeeze(0).transpose(0, 1)  # Now shape: (time, frequency)
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# ----------------------#
# LSTM Model Definition #
# ----------------------#

class LSTMGenreClassifier(nn.Module):
    def __init__(self, input_size=33, hidden_size=128, num_layers=3, num_classes=10, bidirectional=True, dropout=0.3):
        """
        Args:
            input_size (int): Should be equal to n_mfcc (e.g., 13).
            hidden_size (int): Hidden state size.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            bidirectional (bool): Use a bidirectional LSTM.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(LSTMGenreClassifier, self).__init__()
        # input feature normalization
        # applied at each time step
        self.layer_norm_input = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        # output size doubles if bidirectional
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size

        # layer normalization
        self.layer_norm_output = nn.LayerNorm(lstm_out_size)

        self.fc = nn.Linear(lstm_out_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # input normalization
        x = self.layer_norm_input(x)
        out, _ = self.lstm(x)
        # Use the output at the last time step
        last_time_step = out[:, -1, :]  # shape: (batch, lstm_out_size)
        normed_output = self.layer_norm_output(last_time_step)
        out = self.fc(normed_output)
        return out

# -------------------------------#
#     Training and Eval Loop     #
# -------------------------------#

def main():
    data_dir = 'Data/genres_original'
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    target_shape = (128, 150)
    
    # Create dataset and perform an 80/20 train/test split.
    dataset = GenreDataset(data_dir, classes, target_shape)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    split = int(0.8 * dataset_size)
    train_indices, test_indices = indices[:split], indices[split:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create an LSTM model. Input size equals number of mel bands.
    model = LSTMGenreClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # inputs shape: (batch, seq_len, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train
        
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_accuracy = correct_test / total_test
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    torch.save(model.state_dict(), 'model_cache/lstm_genre_classifier.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
