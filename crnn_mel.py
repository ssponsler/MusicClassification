import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader

"""
crnn_mel.py:
Uses CNN layers to extract time-freq features from mel spectrograms,
then feeds time-distributed features into a LSTM for temporal dynamics
modeling.
"""


# -------------------------------#
# Data Preprocessing and Caching #
# -------------------------------#
def load_or_cache_data(data_dir, classes, target_shape=(128, 150), 
                       cache_file='processed_cache/cached_data_mel.pt'):
    """
    If data was preprocessed into a file, load it. Otherwise, process the data.
    """
    if os.path.exists(cache_file):
        print(f"Loading preprocessed data from {cache_file}...")
        data_tensor, labels_tensor = torch.load(cache_file, weights_only=False)
    else:
        print("Preprocessing data...")
        from processing import load_audio_mel
        data_tensor, labels_tensor = load_audio_mel(data_dir, classes, target_shape)
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
            root_dir (str): Directory with genre subfolders.
            classes (list): List of genre names.
            target_shape (tuple): Desired (frequency, time) shape for spectrograms.
            transform (callable, optional): Optional transform applied on each sample.
        """
        self.transform = transform
        self.data, self.labels = load_or_cache_data(root_dir, classes, target_shape)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # shape: (1, frequency, time)
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# -------------------------------#
#           CRNN Model           #
# -------------------------------#
class CRNNGenreClassifier(nn.Module):
    def __init__(self, num_classes=10, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3):
        """
        A CRNN model that uses CNN layers to extract features from mel spectrograms
        and an LSTM to model temporal dependencies.
        
        Args:
            num_classes (int): Number of output classes.
            lstm_hidden_size (int): Hidden state size for the LSTM.
            lstm_num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate used in CNN and LSTM.
        """
        super(CRNNGenreClassifier, self).__init__()
        # --- CNN Front-End ---
        # Input shape: (batch, 1, 128, 150)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))  # reduces both frequency and time
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
        
        # After 3 blocks:
        # Frequency dimension: 128 → 64 → 32 → 16
        # Time dimension: 150 → 75 → 37 → ~18
        # We now want to treat the time dimension as a sequence.
        # We pool only along the frequency dimension to reduce it to 1.
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, None))  # output: (batch, 64, 1, T)
        
        # --- LSTM Back-End ---
        # After pooling, we have (batch, 64, 1, T). Squeeze frequency to get (batch, 64, T)
        # Permute to (batch, T, 64) so that each time step is a 64-dim feature vector.
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_num_layers > 1 else 0)
        lstm_out_size = lstm_hidden_size * 2  # bidirectional
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_size, num_classes)
    
    def forward(self, x):
        # x: (batch, 1, frequency, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # x now: (batch, 64, F, T) where F is around 16 and T is ~18 (depends on input)
        # Pool the frequency dimension to 1:
        x = self.adapt_pool(x)   # shape: (batch, 64, 1, T)
        x = x.squeeze(2)         # shape: (batch, 64, T)
        x = x.permute(0, 2, 1)     # shape: (batch, T, 64)
        
        # LSTM: process the sequence of feature vectors
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, T, hidden*2)
        # Use the output from the final time step:
        x = lstm_out[:, -1, :]     # shape: (batch, hidden*2)
        x = self.dropout(x)
        x = self.fc(x)             # shape: (batch, num_classes)
        return x

# -------------------------------#
#     Training and Eval Loop     #
# -------------------------------#
def main():
    data_dir = 'Data/genres_original'
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    target_shape = (128, 150)  # (frequency, time)
    
    # Create the dataset and an 80/20 train/test split.
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
    model = CRNNGenreClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 50
    start_time = time.time()
    #print(f"Time training start: {start_time}")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
    
    end_time = time.time()
    #print(f"Time training end: {end_time}")
    print(f"Elapsed time: {end_time - start_time} seconds")
    torch.save(model.state_dict(), 'model_cache/crnn_genre_classifier.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
