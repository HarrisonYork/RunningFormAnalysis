import torch.nn as nn
import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class RunningFormDataset(Dataset):
    """
    Custom PyTorch Dataset for loading normalized time-series keypoints.
    """
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tensor_path = row['file_path']
        features = torch.load(tensor_path)

        labels = torch.tensor([
            row['heel_strike'], 
            row['lean_forward'], 
            row['arms_tight'], 
            row['arms_loose']
        ], dtype=torch.float32)
        
        return features, labels


def pad(batch):
    # Separate features and labels from the batch
    features = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    padded_features = pad_sequence(features, batch_first=True)

    cnn_ready_features = padded_features.permute(0, 2, 1)
    return cnn_ready_features, labels


def create_dataloaders(csv_path, batch_size=16):
    """
    Reads the auto-labeled CSV, performs a 70/15/15 split,
    with relative amounts of each error type,
    and returns PyTorch DataLoaders.
    """
    df = pd.read_csv(csv_path)
    
    # Create a unique string identifier for each combination of labels to stratify against
    df['stratify_key'] = df[['heel_strike', 'lean_forward', 'arms_tight', 'arms_loose']].astype(str).agg('_'.join, axis=1)

    # 1. Split off 30% for Temp (Validation + Test) - Stratified
    train_df, temp_df = train_test_split(
        df, test_size=0.30, 
        stratify=df['stratify_key'], 
        random_state=26
    )
    
    # 2. Split the Temp 50/50 into Validation (15%) and Test (15%) - Stratified
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, 
        stratify=temp_df['stratify_key'], 
        random_state=26
    )
    
    print(f"Dataset Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create Datasets
    train_dataset = RunningFormDataset(train_df)
    val_dataset = RunningFormDataset(val_df)
    test_dataset = RunningFormDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)
    
    return train_loader, val_loader, test_loader


def process_results_api(results):
    video_tensor_list = []
    for r in results:
        if len(r.keypoints) == 0 or len(r.boxes) == 0:
            continue

        kpts = r.keypoints.data[0] 
        bbox_height = r.boxes.xywh[0][3] # [x_center, y_center, width, height]
        
        # Left Hip (11) and Right Hip (12)
        l_hip_x, l_hip_y = kpts[11][0], kpts[11][1]
        r_hip_x, r_hip_y = kpts[12][0], kpts[12][1]
        
        hip_center_x = (l_hip_x + r_hip_x) / 2.0
        hip_center_y = (l_hip_y + r_hip_y) / 2.0
        
        normalized_kpts = torch.zeros((17, 3))
        
        for i in range(17):
            x, y, conf = kpts[i]
            if conf > 0:
                normalized_kpts[i][0] = (x - hip_center_x) / bbox_height
                normalized_kpts[i][1] = (y - hip_center_y) / bbox_height
            normalized_kpts[i][2] = conf # visibility unchanged
            
        # flatten the (17, 3) matrix into a 1D vector of length 51
        video_tensor_list.append(normalized_kpts.flatten())

    if len(video_tensor_list) > 0:
        # final 2D PyTorch Tensor (T, 51)
        final_timeseries_tensor = torch.stack(video_tensor_list)
        
    return final_timeseries_tensor


class FormAnalyzer1DCNN(nn.Module):
    def __init__(self, num_features=51, num_classes=4, kernel_size=3):
        super(FormAnalyzer1DCNN, self).__init__()
        
        # 1D Convolutional Layers sliding across the time dimension
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Adaptive pooling handles dynamic time lengths (short clips vs. long clips)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x) # x shape: (Batch, 51, Max_T)
        x = self.global_pool(x)
        x = x.squeeze(-1) 
        x = self.classifier(x)
        return x