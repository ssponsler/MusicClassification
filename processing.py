import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Data Preprocessing Functions
# -------------------------------

def load_audio_mel(directory, classes, target_shape=(128, 150)):
    """
    Loads audio files from the given directory (organized by genre folders),
    computes mel spectrograms on overlapping chunks, and converts each
    spectrogram to a PyTorch tensor of fixed size using adaptive pooling.

    Args:
        directory (str): Root directory containing subdirectories for each genre.
        classes (list): List of class names (subdirectory names).
        target_shape (tuple): Desired (frequency, time) shape of the spectrogram.
                              Default is (128, 150) to match 128 mel bins.

    Returns:
        data_tensor (torch.Tensor): Tensor of shape (N, 1, target_shape[0], target_shape[1])
        labels_tensor (torch.Tensor): Tensor of shape (N,) with class indices.
    """
    data = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load audio at its native sample rate
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    
                    # Define chunking parameters (e.g., 4-second chunks with 2-second overlap)
                    chunk_duration = 4  # seconds
                    overlap_duration = 2  # seconds
                    chunk_samples = int(chunk_duration * sample_rate)
                    overlap_samples = int(overlap_duration * sample_rate)
                    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

                    for i in range(num_chunks):
                        start = i * (chunk_samples - overlap_samples)
                        end = start + chunk_samples
                        chunk = audio_data[start:end]
                        
                        # Pad if the chunk is too short
                        if len(chunk) < chunk_samples:
                            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                        
                        # Compute mel spectrogram (default n_mels=128)
                        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        
                        # Convert the spectrogram to a torch tensor and add channel and batch dimensions:
                        # Shape: (n_mels, time) -> (1, 1, n_mels, time)
                        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        
                        # Use adaptive pooling to force the output size to target_shape.
                        mel_tensor = F.adaptive_avg_pool2d(mel_tensor, output_size=target_shape)
                        
                        # Remove the extra batch dimension: (1, target_shape[0], target_shape[1])
                        mel_tensor = mel_tensor.squeeze(0)
                        
                        data.append(mel_tensor)
                        labels.append(i_class)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    data_tensor = torch.stack(data)  # Shape: (N, 1, target_shape[0], target_shape[1])
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor

def load_audio_lstm(directory, classes, n_mfcc=13, hop_length=512, max_len=128):
    """
    Compute MFCCs for each audio file, pad/truncate each sequence to max_len time steps.
    
    Args:
        directory (str): Root directory containing genre subfolders.
        classes (list): List of genre names.
        n_mfcc (int): Number of MFCC coefficients.
        hop_length (int): Hop length for frame analysis.
        max_len (int): Maximum number of time steps (frames) per sample.
        
    Returns:
        data_tensor (torch.Tensor): Tensor of shape (N, max_len, n_mfcc)
        labels_tensor (torch.Tensor): Tensor of shape (N,)

    """
    data = []
    labels = []
    
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load the audio file (using its native sample rate)
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Extract features:
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)           # (13, T)
                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)  # (1, T)
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)               # (12, T)
                    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length) # (7, T)
                    
                    # Stack along the frequency axis (results in shape (33, T))
                    features = np.vstack([mfcc, spectral_centroid, chroma, spectral_contrast])
                    # Transpose to shape (T, 33): each row is a feature vector for one time frame.
                    features = features.T
                    #features, _, _ = normalize_features(features)
                    
                    # Pad (or truncate) the feature sequence to max_len frames.
                    if features.shape[0] < max_len:
                        pad_width = max_len - features.shape[0]
                        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
                    else:
                        features = features[:max_len, :]
                    
                    data.append(features)
                    labels.append(i_class)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    data = np.stack(data)    # Shape: (N, max_len, 33)
    labels = np.array(labels)
    return data, labels

"""
After experimentation:
Feature normalization for LSTM drastically optimized learning speed,
however was significantly introducing more overfitting which is the main issue
with the model, so this was not used.
Epoch 86/100 -> Loss: 0.0062, Train Acc: 0.9975, Test Acc: 0.2000
"""
def normalize_features(features, mean=None, std=None):
    # features: (T, D)
    if mean is None or std is None:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
    norm_features = (features - mean) / (std + 1e-8)
    return norm_features, mean, std


def load_audio_fused(directory, classes, n_mfcc=13, hop_length=512, max_len=128):
    """
    For each audio file in each genre subfolder, compute:
      - a mel spectrogram (for the CNN branch), and 
      - a stacked spectral feature sequence (for the LSTM branch) 
        including MFCCs, spectral centroid, chroma, and spectral contrast
        (resulting in 33 features per time frame).
    
    Each sample is padded or truncated to a fixed size:
      - Mel spectrogram: (1, mel_target_shape[0], mel_target_shape[1])
      - Spectral features: (max_len, 33)
    
    """
    mel_list = []
    spectral_list = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        print("Processing--", class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # --- CNN Branch: Mel Spectrogram ---
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    # Convert to tensor and add channel dimension → (1, freq, time)
                    mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
                    # Use adaptive pooling to force the output size to mel_target_shape
                    mel_tensor = F.adaptive_avg_pool2d(mel_tensor, output_size=(128,150))
                    
                    # --- LSTM Branch: Spectral Features ---
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)           # (13, T)
                    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)         # (1, T)
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)                      # (12, T)
                    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)         # (7, T)
                    # Stack features → (33, T)
                    features = np.vstack([mfcc, spec_centroid, chroma, spec_contrast])
                    # Transpose to shape (T, 33)
                    features = features.T
                    # Pad or truncate to fixed length max_len
                    if features.shape[0] < max_len:
                        pad_width = max_len - features.shape[0]
                        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
                    else:
                        features = features[:max_len, :]
                    spectral_tensor = torch.tensor(features, dtype=torch.float32)
                    
                    mel_list.append(mel_tensor)           # shape: (1, mel_target_shape[0], mel_target_shape[1])
                    spectral_list.append(spectral_tensor)   # shape: (max_len, 33)
                    labels.append(i_class)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    mel_data = torch.stack(mel_list)             # (N, 1, mel_target_shape[0], mel_target_shape[1])
    spectral_data = torch.stack(spectral_list)   # (N, max_len, 33)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return mel_data, spectral_data, labels_tensor