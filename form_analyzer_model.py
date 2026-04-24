import torch.nn as nn
import torch
import os

def process_results_api(results, filename):
    video_tensor_list = []
    for r in results:
        if len(r.keypoints) == 0 or len(r.boxes) == 0:
            continue

        # Grab the keypoints and bounding box for the FIRST detected person
        # keypoints.data shape is (1, 17, 3) -> [person_index, joint_index, (x, y, conf)]
        kpts = r.keypoints.data[0] 
        bbox_height = r.boxes.xywh[0][3] # [x_center, y_center, width, height]
        
        # Left Hip (11) and Right Hip (12)
        l_hip_x, l_hip_y = kpts[11][0], kpts[11][1]
        r_hip_x, r_hip_y = kpts[12][0], kpts[12][1]
        
        hip_center_x = (l_hip_x + r_hip_x) / 2.0
        hip_center_y = (l_hip_y + r_hip_y) / 2.0
        
        # Initialize an empty tensor for this frame's 17 normalized joints
        normalized_kpts = torch.zeros((17, 3))
        
        # Apply the localized mathematical normalization
        for i in range(17):
            x, y, conf = kpts[i]
            if conf > 0: # Only normalize if the joint is visible
                normalized_kpts[i][0] = (x - hip_center_x) / bbox_height
                normalized_kpts[i][1] = (y - hip_center_y) / bbox_height
            normalized_kpts[i][2] = conf # Keep visibility/confidence score as-is
            
        # Flatten the (17, 3) matrix into a 1D vector of length 51
        video_tensor_list.append(normalized_kpts.flatten())

    # Stack the list of 1D vectors into a final 2D PyTorch Tensor (T, 51)
    if len(video_tensor_list) > 0:
        final_timeseries_tensor = torch.stack(video_tensor_list)
        
    return final_timeseries_tensor


class FormAnalyzer1DCNN(nn.Module):
    def __init__(self, num_features=51, num_classes=4):
        super(FormAnalyzer1DCNN, self).__init__()
        
        # 1D Convolutional Layers sliding across the time dimension
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3), # Regularization to prevent overfitting
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Adaptive pooling handles dynamic time lengths (short clips vs. long clips)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Multi-Label Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 51, Max_T)
        x = self.conv_block(x)
        
        # Crunch down to (Batch, 128, 1) regardless of sequence length
        x = self.global_pool(x)
        
        # Flatten the pool output to (Batch, 128)
        x = x.squeeze(-1) 
        
        # Output is (Batch, 4)
        x = self.classifier(x)
        
        return x