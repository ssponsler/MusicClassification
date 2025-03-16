import os
import numpy as np
import librosa
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Fusion Data Processing Functions
# -------------------------------
def load_or_cache_fusion_data(data_dir, classes, 
                              mel_target_shape=(128,150), 
                              n_mfcc=13, hop_length=512, max_len=128, 
                              cache_file='processed_cache/cached_data_fusion.pt'):
    """
    For each audio file in each genre subfolder, compute:
      - a mel spectrogram (for the CNN branch), and 
      - a stacked spectral feature sequence (for the LSTM branch) 
        including MFCCs, spectral centroid, chroma, and spectral contrast
        (resulting in 33 features per time frame).
    
    Each sample is padded or truncated to a fixed size:
      - Mel spectrogram: (1, mel_target_shape[0], mel_target_shape[1])
      - Spectral features: (max_len, 33)
    
    If a cached file exists, load it. Otherwise, process the data and save it.
    """
    if os.path.exists(cache_file):
        print(f"Loading preprocessed fusion data from {cache_file}...")
        mel_data, spectral_data, labels_tensor = torch.load(cache_file, weights_only=False)
    else:
        print("Preprocessing data...")
        from processing import load_audio_fused
        mel_data, spectral_data, labels_tensor = load_audio_fused(data_dir, classes, n_mfcc=13, 
                                                      hop_length=512, max_len=128)
        print(f"Saving preprocessed data to {cache_file}...")
        torch.save((mel_data, spectral_data, labels_tensor), cache_file)
    
    return mel_data, spectral_data, labels_tensor

# -------------------------------
# Fusion Dataset
# -------------------------------
class FusionGenreDataset(Dataset):
    def __init__(self, data_dir, classes, 
                 n_mfcc=13, hop_length=512, max_len=128, 
                 transform=None):


        self.transform = transform
        self.mel_data, self.spectral_data, self.labels = load_or_cache_fusion_data(
            data_dir, classes, n_mfcc, hop_length, max_len)
    
    def __len__(self):
        return self.mel_data.size(0)
    
    def __getitem__(self, idx):
        mel_sample = self.mel_data[idx]
        spectral_sample = self.spectral_data[idx]
        label = self.labels[idx]
        if self.transform:
            mel_sample = self.transform(mel_sample)
            spectral_sample = self.transform(spectral_sample)
        return mel_sample, spectral_sample, label

# -------------------------------
# CRNN Fusion Model Definition
# -------------------------------
class CRNNFusion(nn.Module):
    def __init__(self, num_classes=10, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3):
        """
        CRNN model that fuses a CNN branch processing mel spectrograms with an LSTM branch
        processing additional spectral features.
        """
        super(CRNNFusion, self).__init__()
        # --- CNN Branch (for mel spectrogram) ---
        # Input: (batch, 1, 128, 150)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2,2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2,2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2,2))
        
        # After conv blocks, assume output shape: (batch, 64, F, T)
        # Use adaptive pooling to collapse frequency dimension to 1 while preserving time
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, None))  # output: (batch, 64, 1, T)
        
        # Global temporal pooling from CNN branch to get a feature vector
        # For instance, average pooling along time to get a 64-dim vector.
        self.cnn_fc = nn.Linear(64, 64)
        
        # --- LSTM Branch (for spectral features) ---
        # Input: (batch, max_len, 33)
        self.lstm = nn.LSTM(input_size=33,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_num_layers > 1 else 0)
        lstm_out_size = lstm_hidden_size * 2
        
        # --- Fusion and Classification ---
        # Fuse CNN and LSTM representations by concatenation.
        self.dropout = nn.Dropout(dropout)
        self.fc_fusion = nn.Linear(64 + lstm_out_size, num_classes)
    
    def forward(self, mel_input, spectral_input):
        # --- CNN Branch ---
        # mel_input: (batch, 1, 128, 150)
        x = F.relu(self.bn1(self.conv1(mel_input)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # x: (batch, 64, F, T) → adaptive pool to (batch, 64, 1, T)
        x = self.adapt_pool(x)
        # Remove frequency dimension: (batch, 64, T)
        x = x.squeeze(2)
        # Global average pooling over time → (batch, 64)
        cnn_repr = torch.mean(x, dim=2)
        cnn_repr = F.relu(self.cnn_fc(cnn_repr))
        
        # --- LSTM Branch ---
        # spectral_input: (batch, max_len, 33)
        lstm_out, _ = self.lstm(spectral_input)  # (batch, max_len, 2*lstm_hidden_size)
        # Use final time step's output as representation → (batch, 2*lstm_hidden_size)
        lstm_repr = lstm_out[:, -1, :]
        
        # --- Fusion ---
        fused = torch.cat((cnn_repr, lstm_repr), dim=1)
        fused = self.dropout(fused)
        logits = self.fc_fusion(fused)
        return logits

# -------------------------------
# Main Training Loop for CRNN Fusion
# -------------------------------
def main():
    data_dir = 'Data/genres_original'
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    mel_target_shape = (128, 150)
    n_mfcc = 13
    hop_length = 512
    max_len = 128
    
    # Create the fusion dataset
    dataset = FusionGenreDataset(data_dir, classes,  
                                 n_mfcc=n_mfcc, hop_length=hop_length, max_len=max_len)
    
    # 80/20 train/test split
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    split = int(0.8 * dataset_size)
    train_indices, test_indices = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset  = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNNFusion(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 50
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for mel_inputs, spectral_inputs, labels in train_loader:
            mel_inputs, spectral_inputs, labels = mel_inputs.to(device), spectral_inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel_inputs, spectral_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * mel_inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train
        
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for mel_inputs, spectral_inputs, labels in test_loader:
                mel_inputs, spectral_inputs, labels = mel_inputs.to(device), spectral_inputs.to(device), labels.to(device)
                outputs = model(mel_inputs, spectral_inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        test_accuracy = correct_test / total_test
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    torch.save(model.state_dict(), 'model_cache/crnn_fusion_genre_classifier.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
