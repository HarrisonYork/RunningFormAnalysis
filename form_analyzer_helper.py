import torch
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
    Reads the auto-labeled CSV, performs a 70/15/15 stratified split,
    and returns PyTorch DataLoaders.
    """
    df = pd.read_csv(csv_path)
    
    # Create a unique string identifier for each combination of labels to stratify against
    df['stratify_key'] = df[['heel_strike', 'lean_forward', 'arms_tight', 'arms_loose']].astype(str).agg('_'.join, axis=1)
    
    # Check for extremely rare combinations (which break stratification)
    # If a combination occurs only once, we duplicate it temporarily or group it
    value_counts = df['stratify_key'].value_counts()
    to_remove = value_counts[value_counts < 3].index
    if len(to_remove) > 0:
        print(f"Warning: Dropping {len(to_remove)} extremely rare label combinations to allow stratification.")
        df = df[~df['stratify_key'].isin(to_remove)]

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