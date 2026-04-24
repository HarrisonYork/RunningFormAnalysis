"""
form_analyzer.py
----------------
Two responsibilities, cleanly separated:
  1. process_results()  — turns raw YOLO output into a normalized .pt tensor
  2. FormAnalyzerCNN    — the model definition used by both training and inference
"""

import os
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)

# ── Where tensors land after processing ──────────────────────────────────────

TENSOR_OUTPUT_DIR = os.path.join(
    "runs", "pose", "user_submissions", "normalized_kpts"
)


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

def process_results(results, filename: str) -> str | None:
    """
    Converts a YOLO results generator into a normalized keypoint tensor and
    saves it to disk.

    Args:
        results:  Generator from model(source=..., stream=True)
        filename: Original video filename (e.g. 'run_001.mp4')

    Returns:
        Path to the saved tensor, or None if no valid frames were found.
    """
    os.makedirs(TENSOR_OUTPUT_DIR, exist_ok=True)
    video_tensor_list = []

    for r in results:
        if len(r.keypoints) == 0 or len(r.boxes) == 0:
            continue

        # First detected person only
        kpts        = r.keypoints.data[0]          # (17, 3) — x, y, conf
        bbox_height = r.boxes.xywh[0][3]           # bounding box height in px

        # Hip-centred normalization anchor
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
            normalized_kpts[i][2] = conf

        video_tensor_list.append(normalized_kpts.flatten())  # → (51,)

    if not video_tensor_list:
        log.warning(f"No valid frames found in {filename}. Skipping save.")
        return None

    tensor = torch.stack(video_tensor_list)  # (T, 51)

    stem = filename.rsplit(".", 1)[0]
    save_path = os.path.join(TENSOR_OUTPUT_DIR, f"{stem}_features.pt")
    torch.save(tensor, save_path)
    log.info(f"Saved tensor {tensor.shape} → {save_path}")
    return save_path


# ── 2. Model ──────────────────────────────────────────────────────────────────

class FormAnalyzerCNN(nn.Module):
    """
    1-D CNN that classifies running form errors from a keypoint time-series.

    Input:  (Batch, T, 51)  — T frames, 51 features per frame
    Output: (Batch, num_classes)  — raw logits (apply sigmoid for probabilities)

    Classes (default):
        0 — overstriding
        1 — forward_lean
        2 — vertical_bounce
    """

    def __init__(self, num_features: int = 51, num_classes: int = 3):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Layer 1 — detect frame-to-frame joint movement
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3),
            # Layer 2 — detect multi-frame gait patterns
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3),
        )

        # Global average pooling → fixed-size representation regardless of T
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 51) → permute to (B, 51, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.adaptive_pool(x)       # (B, 128, 1)
        x = torch.flatten(x, 1)         # (B, 128)
        return self.classifier(x)       # (B, num_classes)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock = torch.randn(4, 90, 51)
    model = FormAnalyzerCNN()
    out = model(mock)
    assert out.shape == (4, 3), f"Unexpected output shape: {out.shape}"
    print(f"✓ Forward pass OK — input {mock.shape} → output {out.shape}")