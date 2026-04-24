import os
import torch
import numpy as np
import pandas as pd

def extract_coords(tensor, joint_idx_1, joint_idx_2=None, axis='x'):
    """Extracts all visible coordinates for a joint or pair of joints."""
    offset = 0 if axis == 'x' else 1
    idx_1, vis_1 = joint_idx_1 * 3 + offset, joint_idx_1 * 3 + 2
    
    vals = []
    for frame in tensor:
        if frame[vis_1] > 0: vals.append(frame[idx_1].item())
        
        if joint_idx_2 is not None:
            idx_2, vis_2 = joint_idx_2 * 3 + offset, joint_idx_2 * 3 + 2
            if frame[vis_2] > 0: vals.append(frame[idx_2].item())
            
    return np.array(vals)

def generate_kinematic_labels(tensor_dir, output_csv="training_labels.csv"):
    labels = []
    
    for file in os.listdir(tensor_dir):
        if not file.endswith('.pt'): continue
            
        filepath = os.path.join(tensor_dir, file)
        tensor = torch.load(filepath) 
        
        # --- 1. Establish Directional Multiplier ---
        # Nose is keypoint 0. Is it in front of the hip (x=0) or behind?
        noses_x = extract_coords(tensor, 0, axis='x')
        if len(noses_x) == 0: continue # Skip if no nose detected
        
        # If mean is positive, they run right (+1). If negative, they run left (-1).
        direction = 1 if np.mean(noses_x) > 0 else -1
        
        # --- Heuristic 1: Overstriding ---
        # Multiply by direction so positive X is ALWAYS the front of the runner
        ankles_x = extract_coords(tensor, 15, 16, axis='x') * direction
        
        # Now we only look at how far FORWARD the ankle gets, ignoring the back leg
        max_front_ankle = np.max(ankles_x) if len(ankles_x) > 0 else 0
        is_overstriding = 1 if max_front_ankle > 0.30 else 0
        
        # --- Heuristic 2: Excessive Forward Lean ---
        shoulders_x = extract_coords(tensor, 5, 6, axis='x') * direction
        avg_shoulder_x = np.mean(shoulders_x) if len(shoulders_x) > 0 else 0
        # If leaning backward, this value is negative. If forward, positive.
        is_leaning = 1 if avg_shoulder_x > 0.15 else 0
        
        # --- Heuristic 3: Excessive Vertical Bounce ---
        # Looking at Y coordinates of the hips (axis='y'). 
        # YOLO Y-axis goes top-to-bottom, but standard deviation handles the absolute spread.
        hips_y = extract_coords(tensor, 11, 12, axis='y')
        vertical_bounce = np.std(hips_y) if len(hips_y) > 0 else 0
        # If the standard deviation of hip height is > 8% of their body height, they are bouncing
        is_bouncing = 1 if vertical_bounce > 0.08 else 0
        
        labels.append({
            "filename": file,
            "overstriding": is_overstriding,
            "forward_lean": is_leaning,
            "vertical_bounce": is_bouncing
        })
        
    df = pd.DataFrame(labels)
    df.to_csv(output_csv, index=False)
    print(f"Successfully auto-labeled {len(df)} sequences with orientation correction!")