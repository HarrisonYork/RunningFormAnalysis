# Attributions

This project was made possible through the use of open-source frameworks, academic datasets, and AI-assisted development tools. Below is a detailed attribution of the resources utilized to build this running form analysis pipeline.

### Ultralytics YOLO11 (Pose Estimation)
The core computer vision pipeline relies on Ultralytics YOLO11 for tracking and extracting frame-by-frame joint keypoints from human movement.
* **YOLO11 Architecture Overview:** [All you need to know about Ultralytics YOLO11](https://www.ultralytics.com/blog/all-you-need-to-know-about-ultralytics-yolo11-and-its-applications)
* **Task Documentation:** [Ultralytics Pose Estimation](https://docs.ultralytics.com/tasks/pose/)

### Gemini
[TODO]

### AthletePose3D
This project references the AthletePose3D dataset for kinematic validation and understanding human pose estimation in complex athletic movements. 

**Citation:**
> Yeung, C., Suzuki, T., Tanaka, R., Yin, Z., & Fujii, K. (2025). *AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements*. arXiv preprint arXiv:2503.07499. [Available here](https://arxiv.org/abs/2503.07499)

**BibTeX:**
```bibtex
@misc{yeung2025athletepose3d,
      title={AthletePose3D: A Benchmark Dataset for 3D Human Pose Estimation and Kinematic Validation in Athletic Movements}, 
      author={Calvin Yeung and Tomohiro Suzuki and Ryota Tanaka and Zhuoer Yin and Keisuke Fujii},
      year={2025},
      eprint={2503.07499},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2503.07499](https://arxiv.org/abs/2503.07499)}, 
}
