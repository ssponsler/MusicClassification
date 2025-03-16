import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------#
#      Dataset Preprocessing     #
# -------------------------------#

def load_or_cache_data(data_dir, classes, target_shape=(128,150), 
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
            root_dir (str): Root directory with subdirectories for each genre.
            classes (list): List of genre names (subdirectory names).
            target_shape (tuple): Desired shape (frequency, time) for the spectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data, self.labels = load_or_cache_data(root_dir, classes, target_shape)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # (1, freq, time)
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# -------------------------------#
#       CNN Model Definition     #
# -------------------------------#

class CNNGenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNGenreClassifier, self).__init__()

        # block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 0.3 dropout layer
        self.dropout = nn.Dropout(0.3)
        self.adapt_pool = nn.AdaptiveAvgPool2d((16, 16))

        # fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # block 1: conv, batch normalization, relu, pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.adapt_pool(x)

        # flatten                
        x = x.view(x.size(0), -1)                 
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------------#
#     Training and Eval Loop     #
# -------------------------------#

def main():
    data_dir = 'Data/genres_original'
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    target_shape = (128, 150)  # (frequency bins, time frames)
    
    dataset = GenreDataset(data_dir, classes, target_shape)
    
    # 80/20 train/test split
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()
    split = int(0.8 * dataset_size)
    train_indices, test_indices = indices[:split], indices[split:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Adam optimizer, using CrossEntropyLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNGenreClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 30
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
    
    # saving model into local file
    torch.save(model.state_dict(), 'model_cache/cnn_classifier.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
