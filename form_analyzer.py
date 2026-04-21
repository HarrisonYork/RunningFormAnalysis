import os
import glob
import numpy as np
import torch
import torch.nn as nn

def process_results(results, filename):
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
        
        tensor_save_path = os.path.join("runs", "pose", "user_submissions", "normalized_kpts", f"{filename.rsplit('.', 1)[0]}_features.pt")
        torch.save(final_timeseries_tensor, tensor_save_path)
        print(f"Successfully saved normalized feature tensor of shape {final_timeseries_tensor.shape}")


class FormAnalyzerCNN(nn.Module):
    def __init__(self, num_features=51, num_classes=3):
        """
        num_features: 51 (17 keypoints * 3 values [x, y, visibility])
        num_classes: Number of form errors we are classifying (e.g., Heel Strike, Overstriding, Posture)
        """
        super(FormAnalyzerCNN, self).__init__()
        
        # Feature Extraction: Sliding across the time dimension
        self.conv_block = nn.Sequential(
            # Layer 1: Detect local frame-to-frame changes
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3), # Regularization for the rubric!
            
            # Layer 2: Detect larger patterns over multiple frames
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3)
        )
        
        # Adaptive pooling ensures the output is always the same size 
        # even if users upload videos of different lengths (different T)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            # Output layer: Raw logits (No sigmoid here, we will use BCEWithLogitsLoss)
            nn.Linear(64, num_classes) 
        )

    def forward(self, x):
        # x input shape from YOLO pipeline: (Batch, T, 51)
        
        # 1. Permute to match PyTorch Conv1d expectations: (Batch, 51, T)
        x = x.permute(0, 2, 1)
        
        # 2. Pass through Convolutional layers
        x = self.conv_block(x)
        
        # 3. Pool to a fixed size and flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # 4. Classify
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Simulate a batch of 4 videos, each 90 frames long, with 51 features
    mock_data = torch.randn(4, 90, 51) 
    model = FormAnalyzerCNN(num_features=51, num_classes=3)
    output = model(mock_data)
    print(f"Input shape: {mock_data.shape}")
    print(f"Output shape: {output.shape}") # Should be [4, 3]